from typing import Literal, Optional, Union
from collections.abc import MutableMapping
from functools import cached_property
from os import path
import ctypes
from warnings import warn

import numpy as np
# FIXME: import only when needed
import h5py

# For simplicity, use AnnData read_elem/write_elem
from anndata._io.specs import read_elem, write_elem
from anndata._core.index import _normalize_indices
from anndata.compat import H5Array, H5Group, ZarrArray, ZarrGroup
from anndata import AnnData

from .elemshadow import ElemShadow, _get_backend_reader

# FIXME: in anndata._types now
ArrayStorageType = Union[ZarrArray, H5Array]
GroupStorageType = Union[ZarrGroup, H5Group]
StorageType = Union[ArrayStorageType, GroupStorageType]


class DataShadow:
    def __init__(
        self,
        filepath,
        array_backend: str = "numpy",
        table_backend: str = "pandas",
        mode: str = "r",
        format: Literal["hdf5", "zarr"] = "hdf5",
    ):
        if format == "zarr":
            import zarr

        if path.exists(filepath):
            if format == "zarr":
                self.file = zarr.open(filepath, mode=mode)
                self.root = "/"
            else:
                # fallback to hdf5 by default
                if format != "hdf5":
                    warn(
                        f"Falling back to hdf5, provided format is '{format}' and not 'hdf5 or 'zarr'"
                    )
                self.file = h5py.File(filepath, mode=mode)
                self.root = "/"
        else:
            file, root = filepath, "/"
            file_exists = False
            i = 1
            if not isinstance(filepath, str):
                filepath = str(filepath)
            while not file_exists and i <= filepath.count("/"):
                path_elements = list(
                    map(lambda x: x[::-1], filepath[::-1].split("/", i))
                )
                filename, root = path_elements[-1], path.join(*path_elements[:-1][::-1])
                file_exists = path.exists(filename)
                i += 1
            if file_exists:
                if format == "zarr":
                    self.file = zarr.open(filepath, mode=mode)
                else:
                    # fallback to hdf5 by default
                    if format != "hdf5":
                        warn(
                            f"Falling back to hdf5, provided format is '{format}' and not 'hdf5 or 'zarr'"
                        )
                    self.file = h5py.File(filename, mode=mode)
                self.root = root
                # Maybe prepend /mod to the modality name
                if root not in self.file and f"/mod/{root}" in self.file:
                    self.root = f"/mod/{root}"
            else:
                raise FileNotFoundError(f"File {filepath} does not seem to exist")
        self._array_backend = array_backend
        self._table_backend = table_backend
        self._ids = {"self": id(self)}
        self._format = "zarr" if format == "zarr" else "hdf5"

        # View-related attributes
        self.is_view = False
        self._oidx = None
        self._vidx = None

    @classmethod
    def _init_as_view(cls, shadow, oidx, vidx):
        if shadow._format == "zarr":
            filename = shadow.file.store.path
            mode = "r+" if not shadow.file.read_only else "r"
        else:
            filename = shadow.file.filename
            mode = shadow.file.mode

        if shadow.root != "/":
            filename = path.join(filename, shadow.root)
        view = DataShadow(
            filename,
            array_backend=shadow._array_backend,
            table_backend=shadow._table_backend,
            mode=mode,
            format=shadow._format,
        )

        # NOTE: Cache is not preserved in a new object

        view.is_view = True
        view._ref = shadow
        view._oidx = oidx
        view._vidx = vidx

        if shadow.is_view:
            view._ref = shadow._ref
            if shadow._oidx is not None:
                if isinstance(shadow._oidx, slice):
                    r = range(*shadow._oidx.indices(shadow._ref.n_obs)).__getitem__(
                        oidx
                    )
                    view._oidx = slice(r.start, r.stop, r.step)
                else:
                    view._oidx = shadow._oidx[oidx]
            if shadow._vidx is not None:
                if isinstance(shadow._vidx, slice):
                    r = range(*shadow._vidx.indices(shadow._ref.n_vars)).__getitem__(
                        vidx
                    )
                    view._vidx = slice(r.start, r.stop, r.step)
                else:
                    view._vidx = shadow._vidx[vidx]

        return view

    @cached_property
    def _obs(self):
        # Use anndata v0.8 spec reader
        reader = _get_backend_reader(self._table_backend)
        obs = self.file[self.root]["obs"]
        columns = {}

        # Deal with legacy
        if isinstance(obs, ArrayStorageType):
            if self._table_backend == "pandas":
                from pandas import DataFrame

                # FIXME: categorical columns?
                table = DataFrame(read_elem(obs))

                if self.is_view:
                    return table.__getitem__(self._oidx)

                return table
            else:
                raise NotImplementedError(
                    "Alternative backends are not available "
                    "for the legacy AnnData/MuData specification."
                )

        if self._table_backend == "pandas":
            table = read_elem(obs)

            if self.is_view:
                # return table.__getitem__(self._oidx)
                return table.iloc[self._oidx]

            return table

        # else (only for AnnData >=0.8)
        for key, value in obs.items():
            col = read_elem(value)
            # Patch categorical parsing for polars
            if self._table_backend == "polars":
                if (
                    "encoding-type" in value.attrs
                    and value.attrs["encoding-type"] == "categorical"
                ):
                    import polars as pl

                    col = pl.Series(col.astype(str)).cast(pl.Categorical)
            columns[key] = col

        table = reader(columns)

        if self.is_view:
            return table.__getitem__(self._oidx)

        return table

    @property
    def obs(self):
        return self._obs

    @cached_property
    def _var(self):
        # Use anndata v0.8 spec reader
        reader = _get_backend_reader(self._table_backend)
        var = self.file[self.root]["var"]
        columns = {}

        # Deal with legacy
        if isinstance(var, ArrayStorageType):
            if self._table_backend == "pandas":
                from pandas import DataFrame

                # FIXME: categorical columns?
                table = DataFrame(read_elem(var))
                if self.is_view:
                    return table.__getitem__(self._vidx)

                return table
            else:
                raise NotImplementedError(
                    "Alternative backends are not available "
                    "for the legacy AnnData/MuData specification."
                )

        if self._table_backend == "pandas":
            table = read_elem(var)

            if self.is_view:
                return table.__getitem__(self._vidx)

            return table

        # else
        for key, value in var.items():
            col = read_elem(value)
            # Patch categorical parsing for polars
            if self._table_backend == "polars":
                if (
                    "encoding-type" in value.attrs
                    and value.attrs["encoding-type"] == "categorical"
                ):
                    import polars as pl

                    col = pl.Series(col.astype(str)).cast(pl.Categorical)
            columns[key] = col

        table = reader(columns)

        if self.is_view:
            return table.__getitem__(self._vidx)

        return table

    @property
    def var(self):
        return self._var

    @cached_property
    def _obs_names(self):
        obs = self.file[self.root]["obs"]

        # Handle legacy
        if isinstance(obs, ArrayStorageType):
            obs_df = self.obs
            if "index" in obs_df.columns:
                return obs_df["index"].values
            elif len(obs_df.columns) > 0:
                index = obs_df.columns[0]
                return obs_df[index].values
            else:
                raise ValueError("Empty obs_names")

        index = "_index"
        if "_index" in obs.attrs:
            index = obs.attrs["_index"]

        if self.is_view:
            return self.file[self.root]["obs"][index][:][self._oidx]

        return self.file[self.root]["obs"][index][:]

    @property
    def obs_names(self):
        return self._obs_names

    @cached_property
    def _var_names(self):
        var = self.file[self.root]["var"]

        # Handle legacy
        if isinstance(var, ArrayStorageType):
            var_df = self.var
            if "index" in var_df.columns:
                return var_df["index"].values
            elif len(var_df.columns) > 0:
                index = var_df.columns[0]
                return var_df[index].values
            else:
                raise ValueError("Empty var_names")

        index = "_index"
        if "_index" in var.attrs:
            index = var.attrs["_index"]
        if self.is_view:
            return self.file[self.root]["obs"][index][:][self._vidx]
        return self.file[self.root]["var"][index][:]

    @property
    def var_names(self):
        return self._var_names

    @cached_property
    def _n_obs(self):
        obs = self.file[self.root]["obs"]
        if isinstance(obs, ArrayStorageType):
            n_obs = obs.shape[0]
        else:
            index = "_index"
            if "_index" in obs.attrs:
                index = obs.attrs["_index"]

            n_obs = obs[index].shape[0]

        if self.is_view and self._oidx is not None:
            if isinstance(self._oidx, slice):
                return len(range(n_obs).__getitem__(self._oidx))
            else:
                return len(self._oidx)
        return n_obs

    @property
    def n_obs(self):
        return self._n_obs

    @cached_property
    def _n_vars(self):
        var = self.file[self.root]["var"]
        if isinstance(var, ArrayStorageType):
            n_vars = var.shape[0]

        else:
            index = "_index"
            if "_index" in var.attrs:
                index = var.attrs["_index"]

            n_vars = var[index].shape[0]

        if self.is_view and self._vidx is not None:
            if isinstance(self._vidx, slice):
                return len(range(n_vars).__getitem__(self._vidx))
            else:
                return len(self._vidx)

        return n_vars

    @property
    def n_vars(self):
        return self._n_vars

    @property
    def shape(self):
        return self.n_obs, self.n_vars

    @cached_property
    def _obsm(self):
        group_storage = (
            self.file[self.root]["obsm"] if "obsm" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            key=path.join(self.root, "obsm"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(self._oidx, None),
        )

    @property
    def obsm(self):
        return self._obsm

    @cached_property
    def _varm(self):
        group_storage = (
            self.file[self.root]["varm"] if "varm" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            # self.file[self.root]["varm"],
            key=path.join(self.root, "varm"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(None, self._vidx),
        )

    @property
    def varm(self):
        return self._varm

    @cached_property
    def _obsp(self):
        group_storage = (
            self.file[self.root]["obsp"] if "obsp" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            # self.file[self.root]["obsp"],
            key=path.join(self.root, "obsp"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(self._oidx, self._oidx),
        )

    @property
    def obsp(self):
        return self._obsp

    @cached_property
    def _varp(self):
        # if "varp" not in self.file[self.root]:
        #    return EmptySlot()
        group_storage = (
            self.file[self.root]["varp"] if "varp" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            # self.file[self.root]["varp"],
            key=path.join(self.root, "varp"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(self._vidx, self._vidx),
        )

    @property
    def varp(self):
        return self._varp

    @cached_property
    def _uns(self):
        if "uns" not in self.file[self.root]:
            return dict()

        def map_get_keys(root):
            s = ElemShadow(
                root,
                key=root.name,
                cache=self.__dict__,
                n_obs=None,
                n_vars=None,
                array_backend=self._array_backend,
                table_backend=self._table_backend,
            )
            for key in root.keys():
                # if hasattr(root[key], "keys"):
                if isinstance(root[key], h5py.Group) and hasattr(root[key], "keys"):
                    s[key] = map_get_keys(root[key])
            return s

        uns_root = self.file[self.root]["uns"]
        return map_get_keys(uns_root)

    @property
    def uns(self):
        return self._uns

    def clear_cache(self):
        keys = list(self.__dict__.keys())
        slots = [
                    "X",
                    "obs",
                    "obsm",
                    "var",
                    "varm",
                    "obsp",
                    "varp",
                    "layers",
                    "raw",
                    "uns",
                ]
        _slots = [f"_{slot}" for slot in slots]
        for key in keys:
            if (
                key.startswith("/")
                or key.startswith("mod/")
                or key in _slots
                or key in slots
            ):
                obj_id = id(self.__dict__[key])
                obj = ctypes.cast(obj_id, ctypes.py_object).value

                del self.__dict__[key]

                # Make sure the object is deleted to free the memory
                del obj

    def close(self):
        if self._format == "zarr":
            self.file.store.close()
            return

        self.file.close()

    def reopen(self, mode: str, file: Optional[str] = None):
        if self._format == "zarr":
            import zarr

        if not self.file:
            if file is None:
                raise ValueError(
                    "The connection is closed but no new file name is provided."
                )
            self.close()
            if self._format == "zarr":
                self.file = zarr.open(file, mode=mode)
            else:
                self.file = h5py.File(file, mode=mode)
        elif self._format == "zarr":
            if (
                self.file.read_only
                and mode != "r"
                or mode == "r"
                and not self.file.read_only
            ):
                file = file or self.file.store.path
                self.close()
                self.file = zarr.open(file, mode=mode)
        elif mode != self.file.mode:
            file = file or self.file.filename
            self.close()
            self.file = h5py.File(file, mode=mode)
        else:
            return self

        # Update ._group in all elements
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"]:
            if key in ["obs", "var"]:
                # In the current implementation attributes are not ElemShadows
                pass
            elif hasattr(self, key):
                elem = getattr(self, key)
                if isinstance(elem, ElemShadow):
                    elem._update_group(self.file[path.join(self.root, key)])

        return self

    def __repr__(self):
        s = ""
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]:
            key_cached = key in self.__dict__
            key_cached_str = RUNECACHED if key_cached else ""

            if key in ["obs", "var"]:
                if key in self.__dict__:
                    s += f"{key}{key_cached_str}:\t"
                    s += f"{', '.join(map(str, getattr(self, key).columns))}\n"
                else:
                    try:
                        key_elems = self.file[self.root][key].keys()
                    except AttributeError:
                        # Do not extract column names from the pre-0.8 AnnData
                        key_elems = ["..."]
                    if len(key_elems) > 0:
                        s += f"{key}:\t{', '.join(key_elems)}\n"
            else:  # complex keys
                if not (key == "uns" and len(self.uns) == 0):
                    # do not show empty dict
                    s += getattr(self, key).__repr__()

        return s

    # Views

    def __getitem__(self, index):
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        return DataShadow._init_as_view(self, oidx, vidx)

    # Legacy methods for scanpy compatibility

    def _sanitize(self):
        pass

    def obs_vector(self, key: str, layer: Optional[str] = None):
        return self.obs[key].values

    def var_vector(self, key: str, layer: Optional[str] = None):
        return self.var[key].values

    # Writing

    def _push_changes(self, clear_cache: bool = False):
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"]:
            if hasattr(self, key):
                elem = getattr(self, key)
                if isinstance(elem, ElemShadow):
                    elem._push_changes(
                        clear_cache=clear_cache,
                    )

    def write(self, *args, **kwargs):
        if self.is_view:
            raise ValueError("Views cannot write data to the file.")
        if (
            self._format == "zarr"
            and self.file.read_only
            or self._format == "hdf5"
            and self.file.mode == "r"
        ):
            raise OSError(
                "File is open in read-only mode. Changes can't be pushed. "
                "Reopen it with .reopen('r+') to enable writing."
            )
        else:
            self._push_changes(*args, **kwargs)
        return self
