from functools import cached_property
from pathlib import Path

import numpy as np
from anndata import AnnData

# For simplicity, use AnnData read_elem/write_elem
from anndata._core.index import _normalize_indices

from .datashadow import DataShadow
from .elemshadow import ElemShadow, RawElemShadow, _get_backend_reader

RUNECACHED = "\u1401"
RUNECACHEDALT = "\u25bc"
RUNENEW = "\u25b2"


class AnnDataShadow(DataShadow):
    def __init__(self, filepath, *args, **kwargs):
        super().__init__(filepath, *args, **kwargs)

    @classmethod
    def _init_as_view(cls, shadow, oidx, vidx):
        if shadow._format == "zarr":
            filename = shadow.file.store.path
            mode = "r+" if not shadow.file.read_only else "r"
        elif shadow._format == "parquet":
            filename = shadow.file.path
            mode = "r+"  # FIXME
            # raise NotImplementedError("Parquet format is not supported for views.")
        else:
            filename = shadow.file.filename
            mode = shadow.file.mode

        if shadow.root != "/":
            filename = (
                str(Path(filename) / shadow.root[1:])
                if shadow.root.startswith("/")
                else str(Path(filename) / shadow.root)
            )
        view = AnnDataShadow(
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
                if isinstance(shadow._oidx, slice) and isinstance(oidx, int | np.integer | slice):
                    r = range(*shadow._oidx.indices(shadow._ref.n_obs)).__getitem__(oidx)
                    if isinstance(r, int | np.integer):
                        view._oidx = np.array([r])
                    view._oidx = slice(r.start, r.stop, r.step)
                elif isinstance(shadow._oidx, slice):
                    view._oidx = np.arange(*shadow._oidx.indices(shadow._ref.n_obs))[oidx]
                else:
                    view._oidx = shadow._oidx[oidx]
            if shadow._vidx is not None:
                if isinstance(shadow._vidx, slice) and isinstance(vidx, int | np.integer | slice):
                    r = range(*shadow._vidx.indices(shadow._ref.n_vars)).__getitem__(vidx)
                    if isinstance(r, int | np.integer):
                        view._vidx = np.array([r])
                    else:
                        view._vidx = slice(r.start, r.stop, r.step)
                elif isinstance(shadow._vidx, slice):
                    view._vidx = np.arange(*shadow._vidx.indices(shadow._ref.n_vars))[vidx]
                else:
                    view._vidx = shadow._vidx[vidx]

        return view

    @cached_property
    def _X(self):
        reader = _get_backend_reader(self._array_backend, self._lazy)
        if self.is_view:
            if (
                isinstance(self._vidx, slice)
                and self._vidx.start is None
                and self._vidx.stop is None
            ):
                x = reader(self.file[self.root]["X"][self._oidx])
            elif (
                isinstance(self._oidx, slice)
                and self._oidx.start is None
                and self._oidx.stop is None
            ):
                x = reader(self.file[self.root]["X"][:, self._vidx])
            else:
                # Only one indexing array at a time is possible
                x = reader(self.file[self.root]["X"][self._oidx][:, self._vidx])
        else:
            x = reader(self.file[self.root]["X"])
        self._ids["X"] = id(x)
        return x

    @property
    def X(self):
        return self._X

    @cached_property
    def _layers(self):
        group_storage = (
            self.file[self.root]["layers"] if "layers" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            key=str(Path(self.root) / "layers"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(self._oidx, self._vidx),
        )

    @property
    def layers(self):
        return self._layers

    @cached_property
    def _raw(self):
        """
        Legacy support. New objects should not use .raw.
        """
        if "raw" in self.file[self.root]:
            group_storage = self.file[self.root]["raw"]
        else:
            group_storage = dict()

        return RawElemShadow(
            group_storage,
            key=str(Path(self.root) / "raw"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=None,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            file=self.file,
            is_view=self.is_view,
            idx=(self._oidx, None),
        )

    @property
    def raw(self):
        return self._raw

    def __repr__(self):
        if self.is_view:
            if self._ref is not None:
                s = f"View of AnnData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars} (original {self._ref.n_obs} × {self._ref.n_vars})\n"
            else:
                s = f"View of AnnData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
        else:
            s = f"AnnData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"

        # X
        key_cached = "X" in self.__dict__ or "_X" in self.__dict__
        key_cached_str = RUNECACHED if key_cached else ""
        if key_cached:
            if "X" in self._ids and self._ids["X"] != id(self.X):
                key_cached_str = RUNECACHEDALT
            elif "_X" in self._ids and self._ids["_X"] != id(self.X):
                key_cached_str = RUNECACHEDALT

        s += f"  X {key_cached_str} \n"

        # raw
        if self.raw and len(self.raw.keys()) > 0:
            s += "  " + self.raw.__repr__()

        # layers
        if len(self.layers) > 0:
            s += "  " + self.layers.__repr__()

        s += "\n".join(["  " + line for line in super().__repr__().strip().split("\n")]) + "\n"

        return s

    def obs_vector(self, key: str, layer: str | None = None):
        if key not in self.obs.columns and key not in self.var_names:
            key = str.encode(key)
        if key in self.var_names:
            # Assume unique var_names
            key_i = np.where(self.var_names == key)[0][0]
            if layer is not None:
                return self.layers[layer][:, key_i]
            else:
                return self.X[:, key_i]

        return self.obs[key].values

    def var_vector(self, key: str, layer: str | None = None):
        if key not in self.var.columns and key not in self.obs_names:
            key = str.encode(key)
        if key in self.obs_names:
            # Assume unique obs_names
            key_i = np.where(self.obs_names == key)[0][0]
            if layer is not None:
                return self.layers[layer][key_i, :]
            else:
                return self.X[key_i, :]

        return self.var[key].values

    # Views

    def __getitem__(self, index):
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        return AnnDataShadow._init_as_view(self, oidx, vidx)

    #
    # It is either this or duck typing.
    #
    # Frequently used tools like scanpy
    # check if the object is an AnnData instance
    # inside quite a few functions.
    #
    # Until those instances are replaced with duck typing,
    # the remedy is to mock the class name.
    #

    @property
    def __class__(self):
        return AnnData
