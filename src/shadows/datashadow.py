import ctypes
import logging
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Literal, get_args
from warnings import warn

# FIXME: import only when needed
import h5py
from anndata._core.index import _normalize_indices

# For simplicity, use AnnData read_elem/write_elem
from anndata.compat import H5Array, H5Group, ZarrArray, ZarrGroup

from .compat import PqArray, PqGroup, read_elem
from .elemshadow import ElemShadow, _get_backend_reader

# FIXME: in anndata._types now
ArrayStorageType = ZarrArray | H5Array | PqArray
GroupStorageType = ZarrGroup | H5Group | PqGroup
StorageType = ArrayStorageType | GroupStorageType


RUNECACHED = "\u1401"
FORMAT_MAP = {
    "h5": "hdf5",
    "hdf5": "hdf5",
    "zarr": "zarr",
    "pq": "parquet",
    "pqdata": "parquet",
}


class DataShadow:
    def __init__(
        self,
        filepath: PathLike,
        array_backend: str = "numpy",
        table_backend: str = "pandas",
        mode: str = "r",
        format: Literal["hdf5", "zarr", "parquet", "h5", "pq", "pqdata"] | None = None,
        lazy: bool = False,
        parent_format: str | None = None,
    ):
        # unify types
        fpstr = str(filepath)
        if filepath.__class__.__name__ == "OpenFile":
            # OpenFile<'file_path'>
            fpstr = str(filepath.path)
        elif filepath.__class__.__name__ == "FSMap":
            # <fsspec.mapping.FSMap at 0x...>
            fpstr = str(filepath.root)
        fpath = Path(fpstr)

        if format is None:
            logging.info("No format provided, trying to infer from the file extension")
            if fpath.suffix == ".zarr":
                format = "zarr"
            elif fpath.suffix == ".pqdata":
                format = "parquet"
            else:
                # NOTE: prioritizing the file extension over the parent format
                # allows to mix formats, e.g. store modalities in .zarr or .hdf5 files
                if parent_format is not None:
                    format = parent_format
                else:
                    format = "hdf5"

        # map the shorthands to the full names
        format = FORMAT_MAP.get(format, format)

        # Auto-detect the format for nested modalities
        # (e.g. m.zarr/mod/x, m.pqdata/mod/y)
        if "zarr" in fpstr or "pqdata" in fpstr and fpath.suffix not in (".zarr", ".pqdata"):
            i = 1
            while i <= fpstr.count("/"):
                path_elements = list(map(lambda x: x[::-1], fpstr[::-1].split("/", i)))
                filename, root = path_elements[-1], str(
                    Path(path_elements[-2]).joinpath(*path_elements[:-2][::-1])
                )
                if Path(filename).suffix == ".zarr":
                    format = "zarr"
                    break
                elif Path(filename).suffix == ".pqdata":
                    format = "parquet"
                    break
                i += 1

        if format == "hdf5":
            import h5py
        elif format == "zarr":
            import zarr
        elif format == "parquet":
            import pqdata

        if fpath.exists():
            if format == "zarr":
                self.file = zarr.open(fpath, mode=mode)
            elif format == "parquet":
                self.file = pqdata.open(fpath, mode=mode)
            else:
                # fallback to hdf5 by default
                if format != "hdf5":
                    warn(
                        f"Falling back to hdf5, provided format is '{format}' and not 'hdf5' or 'zarr'"
                    )
                self.file = h5py.File(fpath, mode=mode)
            self.root = "/"
        else:
            root = "/"
            file_exists = False
            i = 1
            while not file_exists and i <= fpstr.count("/"):
                path_elements = list(map(lambda x: x[::-1], fpstr[::-1].split("/", i)))
                filename, root = path_elements[-1], str(
                    Path(path_elements[-2]).joinpath(*path_elements[:-2][::-1])
                )
                file_exists = Path(filename).exists()
                i += 1
            if file_exists:
                format = FORMAT_MAP.get(Path(filename).suffix[1:], format)
                if format == "zarr":
                    self.file = zarr.open(filename, mode=mode)
                elif format == "parquet":
                    self.file = pqdata.open(filename, mode=mode)
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
            elif (
                filepath.__class__.__name__ == "BufferedReader"
                or filepath.__class__.__name__ == "OpenFile"
                or filepath.__class__.__name__ == "FSMap"
            ):
                # fsspec support
                fname = filepath
                try:
                    from fsspec.core import OpenFile

                    if isinstance(filepath, OpenFile):
                        fname = filepath.__enter__()
                        self._callback = fname.__exit__()
                except ImportError as e:
                    raise ImportError(
                        "To read from remote storage or cache, install fsspec: pip install fsspec"
                    ) from e

                if format == "zarr":
                    self.file = zarr.open(fname, mode=mode)
                elif format == "parquet":
                    self.file = pqdata.open(fname, mode=mode)
                else:
                    raise NotImplementedError(
                        "Only zarr and parquet formats are supported for remote files. "
                        "HDF5 files have to be downloaded first."
                    )
                self.root = "/"
            else:
                raise FileNotFoundError(f"File {fpstr} does not seem to exist")
        self._array_backend = array_backend
        self._table_backend = table_backend
        self._ids = {"self": id(self)}
        self._format = format

        # View-related attributes
        self._is_view = False
        self._oidx = None
        self._vidx = None

        # Laziness behaviour
        self._lazy = lazy

    @classmethod
    def _init_as_view(cls, shadow, oidx, vidx):
        if shadow._format == "zarr":
            filename = shadow.file.store.path
            mode = "r+" if not shadow.file.read_only else "r"
        elif shadow._format == "parquet":
            raise NotImplementedError("Parquet format is not supported for views.")
        else:
            filename = shadow.file.filename
            mode = shadow.file.mode

        if shadow.root != "/":
            filename = str(Path(filename) / shadow.root)
        view = DataShadow(
            filename,
            array_backend=shadow._array_backend,
            table_backend=shadow._table_backend,
            mode=mode,
            format=shadow._format,
        )

        # NOTE: Cache is not preserved in a new object

        view._is_view = True
        view._ref = shadow
        view._oidx = oidx
        view._vidx = vidx

        if shadow.is_view:
            view._ref = shadow._ref
            if shadow._oidx is not None:
                if isinstance(shadow._oidx, slice):
                    r = range(*shadow._oidx.indices(shadow._ref.n_obs)).__getitem__(oidx)
                    view._oidx = slice(r.start, r.stop, r.step)
                else:
                    view._oidx = shadow._oidx[oidx]
            if shadow._vidx is not None:
                if isinstance(shadow._vidx, slice):
                    r = range(*shadow._vidx.indices(shadow._ref.n_vars)).__getitem__(vidx)
                    view._vidx = slice(r.start, r.stop, r.step)
                else:
                    view._vidx = shadow._vidx[vidx]

        return view

    def _annot(self, axis: Literal["obs", "var", 0, 1]):
        if axis not in ("obs", "var", 0, 1):
            raise ValueError(f"axis must be 'obs' or 'var', not {axis}")

        if isinstance(axis, int):
            axis = "obs" if axis == 0 else "var"

        idx = self._oidx if axis == "obs" else self._vidx

        # Use anndata v0.8 spec reader
        reader = _get_backend_reader(self._table_backend, self._lazy)
        annot = self.file[self.root][axis]
        columns = {}

        # Deal with legacy or parquet files
        # TODO: for legacy files,
        # correct the categories for different backends
        #   categories = {}
        #   if "__categories" in annot:
        #       categories = read_elem(annot["__categories"], _format=self._format)

        if isinstance(annot, get_args(ArrayStorageType)):
            if self._table_backend == "pandas":
                from pandas import DataFrame

                # FIXME: categorical columns?
                table = DataFrame(read_elem(annot, _format=self._format))

                if self.is_view:
                    return table.iloc[idx]

                return table
            elif self._table_backend == "polars":
                import polars as pl

                table = read_elem(annot, _format=self._format, kind="polars")

                if self.is_view:
                    if isinstance(idx, pl.Boolean):
                        return table.filter(idx)
                    return table.__getitem__(idx)

                return table
            elif self._table_backend == "pyarrow":

                table = read_elem(annot, _format=self._format, kind="pyarrow")

                if self.is_view:
                    return table.__getitem__(idx)

                return table
            else:
                raise NotImplementedError(
                    "Alternative backends are not available "
                    "for the legacy AnnData/MuData specification."
                )

        if self._table_backend == "pandas":
            table = read_elem(annot, _format=self._format)

            if self.is_view:
                return table.iloc[idx]

            return table

        # else (only for AnnData >=0.8)
        for key, value in annot.items():
            if key == "__categories":
                continue
            col = read_elem(value, _format=self._format)
            if self._table_backend == "polars":
                if "encoding-type" in value.attrs and value.attrs["encoding-type"] == "categorical":
                    import polars as pl

                    col = pl.Series(col.astype(str)).cast(pl.Categorical)
            else:
                raise NotImplementedError("Alternative backends are not fully supported just yet.")
            columns[key] = col

        table = reader(columns)

        if self.is_view:
            if self._table_backend == "pandas":
                return table.iloc[idx]
            return table.__getitem__(idx)

        return table

    @cached_property
    def _obs(self):
        return self._annot("obs")

    @property
    def obs(self):
        return self._obs

    @cached_property
    def _var(self):
        return self._annot("var")

    @property
    def var(self):
        return self._var

    def __names(self, axis: str):
        """
        Internal method to get the names of the obs or var axis
        """
        assert axis in ["obs", "var"], "axis must be 'obs' or 'var'"

        from pandas import Index

        attr = self.file[self.root][axis]

        # Handle legacy
        if isinstance(attr, get_args(ArrayStorageType)):
            attr_df = getattr(self, axis)
            if hasattr(attr_df, "index"):
                names = attr_df.index
            elif hasattr(attr_df, "column_names"):  # pyarrow
                if "index" in attr_df.column_names:
                    names = Index(attr_df["index"])
                elif "__index_level_0__" in attr_df.column_names:
                    names = Index(attr_df["__index_level_0__"])
                elif hasattr(attr_df, "schema"):
                    if hasattr(attr_df.schema, "metadata") and b"pandas" in attr_df.schema.metadata:
                        import json

                        pd_meta = json.loads(attr_df.schema.metadata[b"pandas"])
                        names = Index(attr_df[pd_meta["index_columns"][0]].to_numpy())
                    else:
                        raise ValueError(f"Empty {axis}_names")
            elif hasattr(attr_df, "columns"):
                if "index" in attr_df.columns:
                    names = Index(attr_df["index"])
                elif "__index_level_0__" in attr_df.columns:
                    names = Index(attr_df["__index_level_0__"])
                else:
                    from pyarrow import parquet as pq

                    # TODO: Refactor e.g. by implementing read_elem_schema
                    filename = self.file[self.root][axis].path
                    schema = pq.read_schema(filename)

                    import json

                    try:
                        pd_meta = json.loads(schema.metadata[b"pandas"])
                    except KeyError as e:
                        raise KeyError(f"Metadata from pandas not found in the schema: {e}")

                    names = Index(attr_df[pd_meta["index_columns"][0]])
            else:
                raise ValueError(f"Empty {axis}_names")

        else:
            index = "_index"
            if "_index" in attr.attrs:
                index = attr.attrs["_index"]

            try:
                if self.is_view:
                    indices = self._oidx if axis == "obs" else self._vidx
                    names = Index(self.file[self.root][axis][index][:][indices])
                else:
                    names = Index(self.file[self.root][axis][index][:])
            except KeyError:
                index = "__index_level_0__"
                if self.is_view:
                    indices = self._oidx if axis == "obs" else self._vidx
                    names = Index(self.file[self.root][axis][index][:][indices])
                else:
                    names = Index(self.file[self.root][axis][index][:])

        # only string index
        if all(isinstance(e, bytes) for e in names):
            try:
                names = names.str.decode("utf-8")
            except AttributeError:
                pass

        return names

    @cached_property
    def _obs_names(self):
        """
        Note: currently, anndata relies on pd.Index here
        """
        return self.__names("obs")

    @property
    def obs_names(self):
        return self._obs_names

    @cached_property
    def _var_names(self):
        """
        Note: currently, anndata relies on pd.Index here
        """
        return self.__names("var")

    @property
    def var_names(self):
        return self._var_names

    @cached_property
    def _n_obs(self):
        obs = self.file[self.root]["obs"]
        if isinstance(obs, get_args(ArrayStorageType)):
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
                import numpy as np

                if issubclass(self._oidx.dtype.type, np.bool_):
                    return self._oidx.sum()
                else:
                    return len(self._oidx)
        return n_obs

    @property
    def n_obs(self):
        return self._n_obs

    @cached_property
    def _n_vars(self):
        var = self.file[self.root]["var"]
        if isinstance(var, get_args(ArrayStorageType)):
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
                import numpy as np

                if issubclass(self._vidx.dtype.type, np.bool_):
                    return self._vidx.sum()
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
        group_storage = self.file[self.root]["obsm"] if "obsm" in self.file[self.root] else dict()
        return ElemShadow(
            group_storage,
            key=str(Path(self.root) / "obsm"),
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

    def obsm_keys(self) -> list[str]:
        return list(self._obsm.keys())

    @cached_property
    def _varm(self):
        group_storage = self.file[self.root]["varm"] if "varm" in self.file[self.root] else dict()
        return ElemShadow(
            group_storage,
            # self.file[self.root]["varm"],
            key=str(Path(self.root) / "varm"),
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

    def varm_keys(self) -> list[str]:
        return list(self._varm.keys())

    @cached_property
    def _obsp(self):
        group_storage = self.file[self.root]["obsp"] if "obsp" in self.file[self.root] else dict()
        return ElemShadow(
            group_storage,
            # self.file[self.root]["obsp"],
            key=str(Path(self.root) / "obsp"),
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
        group_storage = self.file[self.root]["varp"] if "varp" in self.file[self.root] else dict()
        return ElemShadow(
            group_storage,
            # self.file[self.root]["varp"],
            key=str(Path(self.root) / "varp"),
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
                if isinstance(root[key], get_args(GroupStorageType)) and hasattr(root[key], "keys"):
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
            if key.startswith("/") or key.startswith("mod/") or key in _slots or key in slots:
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

        if hasattr(self, "_callback") and self._callback and callable(self._callback):
            self._callback()

    def reopen(self, mode: str, file: str | None = None) -> None:
        if self._format == "zarr":
            import zarr

        if not self.file:
            if file is None:
                raise ValueError("The connection is closed but no new file name is provided.")
            self.close()
            if self._format == "zarr":
                self.file = zarr.open(file, mode=mode)
            else:
                self.file = h5py.File(file, mode=mode)
        elif self._format == "zarr":
            if self.file.read_only and mode != "r" or mode == "r" and not self.file.read_only:
                file = file or self.file.store.path
                self.close()
                self.file = zarr.open(file, mode=mode)
        elif mode != self.file.mode:
            file = file or self.file.filename
            self.close()
            self.file = h5py.File(file, mode=mode)
        else:
            return

        # FIXME: parquet support

        # Update ._group in all elements
        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers"]:
            if key in ["obs", "var"]:
                # In the current implementation attributes are not ElemShadows
                pass
            elif hasattr(self, key):
                elem = getattr(self, key)
                if isinstance(elem, ElemShadow):
                    elem._update_group(self.file[str(Path(self.root) / key)])

        return

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

    @property
    def is_view(self):
        return self._is_view

    # Legacy methods for scanpy compatibility

    def _sanitize(self):
        pass

    def obs_vector(self, key: str, layer: str | None = None):
        return self.obs[key].values

    def var_vector(self, key: str, layer: str | None = None):
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

    def write(self, *args, **kwargs) -> None:
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
        return

    def reopen_and_write(self, mode: str = "r+", *args, **kwargs) -> None:
        original_mode = self.file.mode
        self.reopen(mode)
        try:
            self.write(*args, **kwargs)
        except Exception as e:
            warn(f"An error occurred while writing the changes:\n{e}")
        finally:
            self.reopen(original_mode)

    # Laziness

    def lazy(self):
        self._lazy = True

    def eager(self):
        self._lazy = False

    @property
    def is_lazy(self):
        return self._lazy
