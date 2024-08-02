from functools import cached_property
from pathlib import Path

import numpy as np

# For simplicity, use AnnData read_elem/write_elem
from anndata._core.index import _normalize_indices

from .anndatashadow import AnnDataShadow
from .datashadow import DataShadow
from .elemshadow import ElemShadow


class MuDataShadow(DataShadow):
    def __init__(self, filepath, *args, **kwargs):
        super().__init__(filepath, *args, **kwargs)
        mods = list(self.file["mod"].keys())

        modorder = mods
        if "mod-oder" in self.file["mod"].attrs:
            modorder_raw = self.file["mod"].attrs["mod-order"]
            if all(m in mods for m in modorder_raw):
                modorder = [m for m in modorder_raw if m in mods]

        kwargs["parent_format"] = self._format
        try:
            self.mod = {
                k: AnnDataShadow(Path(filepath) / "mod" / k, *args, **kwargs) for k in modorder
            }
        except (FileNotFoundError, TypeError) as e:
            # fsspec.mapping.FSMap
            try:
                from fsspec.mapping import FSMap

                if not isinstance(filepath, FSMap):
                    raise NotImplementedError(
                        "remote storage support has only been implemented for FSMap interface"
                    )
                if filepath.fs.__class__.__name__ != "S3FileSystem":
                    raise NotImplementedError(
                        "fsspec.mapping.FSMap has only been implemented for S3FileSystem"
                    )

                mapper = filepath.fs.get_mapper
                self.mod = {
                    k: AnnDataShadow(
                        mapper(str(Path(filepath.root) / "mod" / k)),
                        format=self._format,
                        *args,
                        **kwargs,
                    )
                    for k in modorder
                }
            except Exception:
                raise e

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
        elif shadow._format == "parquet":
            filename = shadow.file.path
            mode = "r+"  # FIXME
        else:
            filename = shadow.file.filename
            mode = shadow.file.mode

        if shadow.root != "/":
            filename = str(Path(filename) / shadow.root)
        view = MuDataShadow(
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
            for attr, idx in (("_oidx", oidx), ("_vidx", vidx)):
                shadow_idx = getattr(shadow, attr)
                if shadow_idx is not None:
                    n_attr = shadow._ref.n_obs if attr == "_oidx" else shadow._ref.n_vars
                    if isinstance(shadow_idx, slice) and isinstance(idx, int | np.integer | slice):
                        r = range(*shadow_idx.indices(n_attr)).__getitem__(idx)
                        if isinstance(r, int | np.integer):
                            setattr(view, attr, np.array([r]))
                        setattr(view, attr, slice(r.start, r.stop, r.step))
                    elif isinstance(shadow_idx, slice):
                        setattr(view, attr, np.arange(*shadow_idx.indices(shadow._ref.n_obs))[idx])
                    elif hasattr(shadow_idx.dtype, "type") and issubclass(
                        shadow_idx.dtype.type, np.bool_
                    ):
                        if hasattr(idx.dtype, "type") and issubclass(idx.dtype.type, np.bool_):
                            view_idx = shadow_idx[shadow_idx] = idx
                            setattr(view, attr, view_idx)
                        else:
                            setattr(view, attr, shadow_idx[np.where(idx)[0]])
                    else:
                        setattr(view, attr, shadow_idx[idx])

        for mod, modality in view.mod.items():
            # Subsetting doesn't depend on axis:
            # axis implicitly influences .obsmap / .varmap
            if isinstance(oidx, slice) and oidx.start is None and oidx.stop is None:
                mod_obs = oidx
            else:
                mod_obs = shadow.obsmap[mod][oidx]
                if hasattr(mod_obs, "columns") and mod in mod_obs.columns:
                    mod_obs = mod_obs[mod].values
                mod_obs = mod_obs[mod_obs != 0] - 1

            if isinstance(vidx, slice) and vidx.start is None and vidx.stop is None:
                mod_vars = vidx
            else:
                mod_vars = shadow.varmap[mod][vidx]
                if hasattr(mod_obs, "columns") and mod in mod_obs.columns:
                    mod_obs = mod_obs[mod].values
                mod_vars = mod_vars[mod_vars != 0] - 1

            view.mod[mod] = modality[mod_obs, mod_vars]
            view.mod[mod]._ref = shadow[mod]
            if hasattr(modality.file, "close") and callable(modality.file.close):
                modality.file.close()

            # TODO: avoid creating a non-view AnnData connection
            # in the MuDataShadow() constructor above

        return view

    @cached_property
    def _obsmap(self):
        group_storage = (
            self.file[self.root]["obsmap"] if "obsmap" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            key=str(Path(self.root) / "obsmap"),
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
            self.file[self.root]["varmap"] if "varmap" in self.file[self.root] else dict()
        )
        return ElemShadow(
            group_storage,
            key=str(Path(self.root) / "varmap"),
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
                elem._update_group(self.file[str(Path(self.root) / key)])

        return self

    def __repr__(self):
        if self.is_view:
            if self._ref is not None:
                s = f"View of MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars} (original {self._ref.n_obs} × {self._ref.n_vars})\n"
            else:
                s = f"View of MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
        else:
            s = f"MuData Shadow object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"

        s += "\n".join(["  " + line for line in super().__repr__().strip().split("\n")]) + "\n"

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
