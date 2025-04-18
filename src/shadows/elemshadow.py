from collections.abc import MutableMapping
from functools import cached_property, partial
from pathlib import Path
from typing import get_args
from warnings import warn

# For simplicity, use AnnData read_elem/write_elem
from anndata._io.specs import write_elem
from anndata.compat import H5Array, H5Group, ZarrArray, ZarrGroup

from .compat import PqArray, PqGroup, read_elem

ArrayStorageType = ZarrArray | H5Array | PqArray
GroupStorageType = ZarrGroup | H5Group | PqGroup
StorageType = ArrayStorageType | GroupStorageType

RUNECACHED = "\u1401"
RUNECACHEDALT = "\u25bc"
RUNENEW = "\u25b2"


class LazyReader:
    def __init__(self, reader, data):
        self.reader = reader
        self.data = data
        self.f = lambda data, slice: reader(data[slice])
        self.partial = partial(self.f, self.data)

    def __call__(self, value):
        return self.partial(value)

    def __getitem__(self, value):
        return self.partial(value)


def _get_backend_reader(backend, lazy: bool = False):
    if callable(backend):
        reader = backend
    else:
        if backend == "numpy":
            import numpy as np

            # TODO: Handle sparsity
            reader = np.array

        elif backend == "jax":
            import jax.numpy as jnp

            reader = jnp.array

        elif backend == "torch" or backend == "pytorch":
            import torch

            reader = torch.Tensor

        elif backend == "pandas":
            import pandas as pd

            reader = pd.DataFrame

        elif backend == "polars":
            import polars as pl

            reader = pl.from_dict

        elif backend == "arrow" or backend == "pyarrow":
            import pyarrow as pa

            reader = pa.Table.from_pydict

        else:
            return NotImplementedError

    if lazy:
        base_reader = reader

        def reader(data):
            return LazyReader(base_reader, data)

    return reader


class EmptySlot:
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __repr__(self):
        return ""


class ElemShadow(MutableMapping):
    def __init__(
        self,
        group_storage,
        key: str,
        cache: dict | None = None,
        n_obs: int | None = None,
        n_vars: int | None = None,
        array_backend: str = "numpy",
        table_backend: str = "pandas",
        is_view: bool | None = False,
        idx=None,
    ):
        self._group = group_storage
        self._key = key
        self._cache = cache
        self._n_obs = n_obs
        self._n_vars = n_vars

        try:
            self._elems = list(self._group.keys())
        except AttributeError as e:
            # This block below is only to handle legacy files
            # where this can be a structured array.
            # Legacy file support will get deprecated in later versions.
            import numpy as np

            in_memory = np.array(self._group)
            fields = in_memory.dtype.fields
            if fields is not None:
                self._elems = list(fields.keys())
                if self._key not in cache:
                    self._cache[self._key] = dict()
                    for value in self._elems:
                        value_path = str(Path(self._key) / value)
                        value_out = in_memory[value]

                        key_name = Path(self._key).name
                        if is_view:
                            oidx, vidx = idx
                            if self._key.endswith("layers"):
                                if oidx is not None and vidx is not None:
                                    value_out = value_out[oidx, vidx]
                                elif oidx is not None:
                                    value_out = value_out.__getitem__(oidx)
                                elif vidx is not None:
                                    value_out = value_out[:, vidx]
                            elif key_name.startswith("obs"):
                                if oidx is not None:
                                    value_out = value_out.__getitem__(oidx)
                                    if key_name == "obsp":
                                        value_out = value_out[:, oidx]
                            elif key_name.startswith("var"):
                                if vidx is not None:
                                    value_out = value_out.__getitem__(vidx)
                                    if key_name == "varp":
                                        value_out = value_out[:, vidx]

                        self._cache[value_path] = value_out
            else:
                raise AttributeError("Cannot handle this legacy file: " + str(e)) from e

        self._newelems = dict()
        self._nested = dict()

        self._array_backend = array_backend
        self._table_backend = table_backend

        self.is_view = is_view
        self._idx = idx

    def __getitem__(self, value):
        value_path = str(Path(self._key) / value)
        if value_path in self._cache:
            return self._cache[value_path]
        elif value in self._newelems:
            return self._newelems[value]
        else:
            value_elem = self._group[value]
            # is_group = type(value_elem).__name__ == 'Group'  # h5py.Group, zarr.hierarchy.Group
            is_group = isinstance(value_elem, get_args(GroupStorageType))

            # Return the nested ElemShadow
            if value_path in self._nested:
                return self._nested[value_path]

            # Directly read it if it is a scalar dataset
            # NOTE: Sparse matrices and data frames are groups
            elif not is_group and value_elem.shape == ():
                value_out = self._group[value][()]
                if isinstance(value_out, bytes):
                    try:
                        # bytes -> string
                        value_out = value_out.decode()
                    except AttributeError:
                        pass

            elif self._array_backend == "numpy" and self._table_backend == "pandas":
                # HOTFIX
                if self._group[value].__class__.__module__ == "pqdata.core":
                    value_out = read_elem(self._group[value], _format="parquet")
                else:
                    value_out = read_elem(self._group[value])

            else:
                if (
                    "encoding-type" in value_elem.attrs
                    and value_elem.attrs["encoding-type"] == "array"
                ):
                    reader = _get_backend_reader(self._array_backend)
                elif (
                    "encoding-type" in value_elem.attrs
                    and value_elem.attrs["encoding-type"] == "dataframe"
                ):
                    reader = _get_backend_reader(self._table_backend)
                else:
                    reader = _get_backend_reader(self._array_backend)
                # TODO: avoid reading the whole dataset
                if isinstance(self._group, PqGroup):
                    value_out = read_elem(self._group[value], _format="parquet")
                    try:
                        value_out = reader(value_out)
                    except ValueError as e:
                        if hasattr(value_out, "todense") and callable(value_out.todense):
                            value_out = reader(value_out.todense())
                        else:
                            raise e
                else:
                    try:
                        value_out = reader(self._group[value][:])
                    except TypeError:
                        # e.g. sparse matrices
                        value_out = read_elem(self._group[value])
                        try:
                            value_out = reader(value_out)
                        except ValueError as e:
                            if hasattr(value_out, "todense") and callable(value_out.todense):
                                value_out = reader(value_out.todense())
                            else:
                                raise e

            # slicing behaviour depends on the attribute
            key_name = Path(self._key).name
            if self.is_view:
                oidx, vidx = self._idx
                if self._key.endswith("layers"):
                    if oidx is not None and vidx is not None:
                        value_out = value_out[oidx, vidx]
                    elif oidx is not None:
                        value_out = value_out.__getitem__(oidx)
                    elif vidx is not None:
                        value_out = value_out[:, vidx]
                elif key_name.startswith("obs"):
                    if oidx is not None:
                        value_out = value_out.__getitem__(oidx)
                        if key_name == "obsp":
                            value_out = value_out[:, oidx]
                elif key_name.startswith("var"):
                    if vidx is not None:
                        value_out = value_out.__getitem__(vidx)
                        if key_name == "varp":
                            value_out = value_out[:, vidx]

            self._cache[value_path] = value_out
            return value_out

    def __setitem__(self, key, value):
        value_path = str(Path(self._key) / key)

        if self._key.endswith("obsm") or self._key.endswith("obsp") or self._key.endswith("layers"):
            if self._n_obs is None:
                if key in self._elems:
                    self._n_obs = self._group[key].shape[0]

            if self._n_obs is not None:
                assert value.shape[0] == self._n_obs, "Shape mismatch"
                if self._key.endswith("obsp"):
                    assert value.shape[1] == self._n_obs, "Shape mismatch"

        if self._key.endswith("varm") or self._key.endswith("varp") or self._key.endswith("layers"):
            if self._n_vars is None:
                if key in self._elems:
                    self._n_vars = self._group[key].shape[0]

            if self._n_vars is not None:
                if self._key.endswith("layers"):
                    assert value.shape[1] == self._n_vars, "Shape mismatch"
                else:  # varm, varp
                    assert value.shape[0] == self._n_vars, "Shape mismatch"
                    if self._key.endswith("varp"):
                        assert value.shape[1] == self._n_vars, "Shape mismatch"

        if key in self._elems:
            if isinstance(self._group[key], get_args(GroupStorageType)):
                self._nested[value_path] = value
            else:
                self._cache[value_path] = value
        else:
            self._newelems[key] = value

    def __delitem__(self, key):
        if key in self._newelems:
            del self._newelems[key]
        else:
            raise NotImplementedError("Cannot delete data " "that already exists in the file")

    def __contains__(self, value):
        if value in self._elems or value in self._newelems:
            return True
        return False

    def __iter__(self):
        all_keys = self._elems + list(self._newelems.keys())
        for i, key in enumerate(all_keys):
            yield key, self[key]

    def keys(self):
        return self._elems + list(self._newelems.keys())

    def values(self):
        all_keys = self._elems + list(self._newelems.keys())
        for i, key in enumerate(all_keys):
            yield key, self[key]

    def items(self):
        for key in self._elems:
            yield key, self[key]

        for key, value in self._newelems.items():
            yield key, value

    def __len__(self):
        return len(self._elems) + len(self._newelems)

    def __repr__(self):
        s = ""
        key_elems_str, new_elems_str = [], []

        if len(self._elems) > 0:
            key_elems_cached = [str(Path(self._key) / e) in self._cache for e in self._elems]
            key_elems_cached_str = [RUNECACHED if e_cached else "" for e_cached in key_elems_cached]
            # TODO: RUNECACHEDALT
            key_elems_str = list(
                map(lambda xs: "".join(xs), zip(self._elems, key_elems_cached_str))
            )

        if len(self._newelems) > 0:
            new_elems_str = [f"{e}{RUNENEW}" for e in self._newelems.keys()]

        all_elems_str = key_elems_str + new_elems_str
        if len(all_elems_str) > 0:
            s += f"{Path(self._key).name}:\t{', '.join(all_elems_str)}\n"

        return s

    # Writing

    def _push_changes(self, clear_cache: bool = False):
        if len(self._newelems) > 0:
            keys = list(self._newelems.keys())
            for key in keys:
                write_elem(self._group, key, self._newelems[key])
                if not clear_cache:
                    self._cache[str(Path(self._key) / key)] = self._newelems[key]
                del self._newelems[key]
            self._elems = list(self._group.keys())

    def _update_group(self, group):
        self._group = group
        for elem in self._nested.values():
            elem._update_group(group)


class RawElemShadow(ElemShadow):
    def __init__(
        self,
        group_storage,
        key: str,
        file: str,
        cache: dict | None = None,
        n_obs: int | None = None,
        n_vars: int | None = None,
        array_backend: str = "numpy",
        table_backend: str = "pandas",
        is_view: bool = False,
        idx=None,
    ):
        super().__init__(
            group_storage=group_storage,
            key=key,
            cache=cache,
            n_obs=n_obs,
            n_vars=n_vars,
            array_backend=array_backend,
            table_backend=table_backend,
            is_view=is_view,
            idx=idx,
        )
        self.file = file
        self._ids = {"self": id(self)}

    @cached_property
    def _X(self):
        return self.__getitem__("X")

    @property
    def X(self):
        return self._X

    @cached_property
    def _var(self):
        return self.__getitem__("var")

    @property
    def var(self):
        return self._var

    @cached_property
    def _var_names(self):
        index = "_index"
        var = self._group["var"]
        if "_index" in var.attrs:
            index = var.attrs["_index"]
        if self.is_view and len(self._idx) > 1 and self._idx[1] is not None:
            return self._group["var"][index][self._idx[1]]
        return self._group["var"][index][:]

    @property
    def var_names(self):
        return self._var_names

    @cached_property
    def __n_obs(self):
        x = self._group["X"]
        if isinstance(x, get_args(ArrayStorageType)):
            n_obs = x.shape[0]
        else:
            n_obs = x.attrs["shape"][0]

        if self.is_view and self._idx[0] is not None:
            oidx = self._idx[0]
            if isinstance(oidx, slice):
                n_obs = len(range(n_obs).__getitem__(oidx))
            else:
                n_obs = len(oidx)

        return n_obs

    @property
    def n_obs(self):
        if self._n_obs is None:
            return self.__n_obs
        return self._n_obs

    @cached_property
    def __n_vars(self):
        if "var" in self._group:
            var = self._group["var"]
            if isinstance(var, get_args(ArrayStorageType)):
                n_vars = var.shape[0]

            else:
                index = "_index"
                if "_index" in var.attrs:
                    index = var.attrs["_index"]

                n_vars = var[index].shape[0]
        else:
            x = self._group["X"]
            if isinstance(x, get_args(ArrayStorageType)):
                n_vars = x.shape[1]
            else:
                n_vars = x.attrs["shape"][1]

        self._n_vars = n_vars
        return n_vars

    @property
    def n_vars(self):
        if self._n_vars is None:
            return self.__n_vars
        return self._n_vars

    @property
    def shape(self):
        return self.n_obs, self.n_vars

    @cached_property
    def _varm(self):
        storage_group = self._group["varm"] if "varm" in self._elems else dict()
        return ElemShadow(
            storage_group,
            key=str(Path(self._group.name) / "varm"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=self.idx,
        )

    @property
    def varm(self):
        return self._varm

    # No writing: .raw is always read-only

    def _push_changes(self, *args, **kwrags):
        warn("Raw object is always read-only. No changes will be written.")
