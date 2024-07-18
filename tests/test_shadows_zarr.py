from pathlib import Path
import pytest
from typing import Optional

from shadows import AnnDataShadow, MuDataShadow

import numpy as np
from scipy.sparse import coo_matrix
from anndata import AnnData
from mudata import MuData

N, D = 50, 20


def matrix(sparse_x: bool = False, n: Optional[int] = None, d: Optional[int] = None):
    np.random.seed(100)

    if n is None:
        n = N
    if d is None:
        d = D

    if sparse_x:
        sparsity = 0.2
        row = np.random.choice(n, 1000 * sparsity)
        col = np.random.choice(d, 1000 * sparsity)
        data = np.random.normal(size=1000 * sparsity)

        x = coo_matrix((data, (row, col)), shape=(n, d)).tocsr()
    else:
        x = np.random.normal(size=(n, d))
    return x


@pytest.fixture()
def adata(sparse_x: bool = False, obsm: bool = False):
    x = matrix(sparse_x)
    ad = AnnData(X=x)

    return ad


@pytest.fixture()
def mdata(sparse_x: bool = False, sparse_y: bool = False):
    np.random.seed(42)

    xn, xd = np.random.choice(100, 2)
    yn, yd = np.random.choice(100, 2)

    x = matrix(sparse_x, n=xn, d=xd)
    y = matrix(sparse_y, n=yn, d=yd)

    ax = AnnData(X=x)
    ay = AnnData(X=y)

    ax.var_names = [f"x{i}" for i in range(xd)]
    ay.var_names = [f"y{i}" for i in range(yd)]

    mdata = MuData({"x": ax, "y": ay})

    return mdata


@pytest.mark.usefixtures("filepath_mudata_zarr")
class TestMuData:
    def test_mudata_simple(self, mdata, filepath_mudata_zarr):
        filename = filepath_mudata_zarr
        mdata.write_zarr(filename)

        msh = MuDataShadow(filename)

        assert mdata.shape == msh.shape

        msh.close()

    def test_anndata_inside_mudata(self, mdata, filepath_mudata_zarr):
        filename = filepath_mudata_zarr
        mdata.write_zarr(filename)

        mod_x = Path(filename) / "mod" / "x"
        mod_y = Path(filename) / "mod" / "y"

        ash_x = AnnDataShadow(mod_x)
        ash_y = AnnDataShadow(mod_y)

        assert ash_x.shape == mdata["x"].shape
        assert ash_y.shape == mdata["y"].shape

        ash_x.close()
        ash_y.close()
