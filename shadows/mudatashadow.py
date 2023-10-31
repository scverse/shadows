from typing import Optional
from collections.abc import MutableMapping
from functools import cached_property
from os import path
import ctypes
from warnings import warn

import numpy as np
import h5py

# For simplicity, use AnnData read_elem/write_elem
from anndata._io.specs import read_elem, write_elem
from anndata._core.index import _normalize_indices


from .elemshadow import ElemShadow
from .datashadow import DataShadow
from .anndatashadow import AnnDataShadow


class MuDataShadow(DataShadow):
    def __init__(self, filepath, *args, **kwargs):
        super().__init__(filepath, *args, **kwargs)
        mods = list(self.file["mod"].keys())

        modorder = mods
        if "mod-oder" in self.file["mod"].attrs:
            modorder_raw = self.file["mod"].attrs["mod-order"]
            if all(m in mods for m in modorder_raw):
                modorder = [m for m in modorder_raw if m in mods]

        self.mod = {
            k: AnnDataShadow(f"{filepath}/mod/{k}", *args, **kwargs) for k in modorder
        }
        self.n_mod = len(self.mod)
        self.mask = None

        self._axis = 0
        if self.file:
            if "axis" in self.file[self.root].attrs:
                self._axis = self.file[self.root].attrs["axis"]

        # To handle scanpy plotting calls and other tools
        self.raw = None

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
        view = MuDataShadow(
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

        for mod, modality in view.mod.items():
            # Subsetting doesn't depend on axis:
            # axis implicitly influences .obsmap / .varmap
            if isinstance(oidx, slice) and oidx.start is None and oidx.stop is None:
                mod_obs = oidx
            else:
                mod_obs = shadow.obsmap[mod][oidx]
                mod_obs = mod_obs[mod_obs != 0] - 1

            if isinstance(vidx, slice) and vidx.start is None and vidx.stop is None:
                mod_vars = vidx
            else:
                mod_vars = shadow.varmap[mod][vidx]
                mod_vars = mod_vars[mod_vars != 0] - 1

            view.mod[mod] = modality[mod_obs, mod_vars]
            view.mod[mod]._ref = shadow[mod]
            modality.file.close()

            # TODO: avoid creating a non-view AnnData connection
            # in the MuDataShadow() constructor above

        return view

    @cached_property
    def _obsmap(self):
        group_storage = (
            self.file[self.root]["obsmap"]
            if "obsmap" in self.file[self.root]
            else dict()
        )
        return ElemShadow(
            group_storage,
            key=path.join(self.root, "obsmap"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(self._oidx, None),
        )

    @property
    def obsmap(self):
        return self._obsmap

    @cached_property
    def _varmap(self):
        group_storage = (
            self.file[self.root]["varmap"]
            if "varmap" in self.file[self.root]
            else dict()
        )
        return ElemShadow(
            group_storage,
            key=path.join(self.root, "varmap"),
            cache=self.__dict__,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            array_backend=self._array_backend,
            table_backend=self._table_backend,
            is_view=self.is_view,
            idx=(None, self._vidx),
        )

    @property
    def varmap(self):
        return self._varmap

    def clear_cache(self):
        super().clear_cache()
        for modality in self.mod.values():
            modality.clear_cache()

    def close(self, close_modalities: bool = True):
        if close_modalities:
            for modality in self.mod.values():
                modality.close()
        super().close()

    def reopen(self, mode: str):
        if not self.file or mode != self.file.mode:
            file = self.file.filename
            super().reopen(mode=mode)
            for modality in self.mod.values():
                modality.reopen(mode=mode, file=file)
        else:
            return self

        # Update ._group in all elements
        for key in ["mod"]:
            elem = getattr(self, key)
            if isinstance(elem, ElemShadow):
                elem._update_group(self.file[path.join(self.root, key)])

        return self

    def __repr__(self):
        if self.is_view:
            if self._ref is not None:
                s = f"View of MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars} (original {self._ref.n_obs} × {self._ref.n_vars})\n"
            else:
                s = f"View of MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
        else:
            s = f"MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"

        s += (
            "\n".join(["  " + line for line in super().__repr__().strip().split("\n")])
            + "\n"
        )

        # obsmap and varmap
        for k in ["obsmap", "varmap"]:
            item = getattr(self, k)
            if len(item) > 0:
                s += "  " + item.__repr__()

        s += f"  mod:\t{self.n_mod} modalit{'ies' if self.n_mod > 1 else 'y'}\n"
        for m, modality in self.mod.items():
            m_repr = modality.__repr__().strip().split("\n")[1:]
            s += f"    {m}: {modality.n_obs} x {modality.n_vars}\n"
            s += "\n".join(["      " + line for line in m_repr]) + "\n"
        return s

    # Writing

    def _push_changes(self, clear_cache: bool = False):
        super()._push_changes(clear_cache=clear_cache)
        for modality in self.mod.values():
            modality._push_changes(
                clear_cache=clear_cache,
            )

    # Views

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.mod[index]
        oidx, vidx = _normalize_indices(index, self.obs_names, self.var_names)
        return MuDataShadow._init_as_view(self, oidx, vidx)

    #
    # Same as for AnnData above:
    # in the absence of duck typing in most tools,
    # the solution is to mock the class.
    #

    @property
    def __class__(self):
        try:
            from mudata import MuData

            return MuData
        except ModuleNotFoundError:
            return MuDataShadow
