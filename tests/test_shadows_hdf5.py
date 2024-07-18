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


@pytest.mark.usefixtures("filepath_h5ad")
class TestAnnData:
    @pytest.mark.parametrize("sparse_x", [True, False])
    def test_anndata_simple(self, adata, filepath_h5ad, sparse_x):
        filename = filepath_h5ad
        adata.write(filename)

        ash = AnnDataShadow(filename)

        assert adata.shape == ash.shape

        ash.close()

    def test_anndata_obs(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_obs.h5ad")

        adata.obs["logical"] = np.random.choice([True, False], size=N)
        adata.obs["integers"] = np.arange(N)
        adata.obs["floats"] = np.random.normal(size=N)
        adata.obs["strings"] = np.random.choice(["abc", "def"], size=N)
        adata.obs["categories"] = adata.obs["strings"].astype("category")

        adata.write(filename)

        ash = AnnDataShadow(filename)

        for key in ["logical", "integers", "floats", "strings", "categories"]:
            assert key in ash.obs.columns
            assert ash.obs[key].equals(adata.obs[key])

        assert adata.obs.shape == ash.obs.shape

    def test_anndata_obsm(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_obsm.h5ad")

        for i in range(2, 10):
            adata.obsm["X_test"] = np.random.normal(size=(N, 2))
            adata.write(filename)

            ash = AnnDataShadow(filename)

            assert "X_test" in ash.obsm
            assert adata.obsm["X_test"].shape == ash.obsm["X_test"].shape

            ash.close()

    def test_anndata_var(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_var.h5ad")

        adata.var["logical"] = np.random.choice([True, False], size=D)
        adata.var["integers"] = np.arange(D)
        adata.var["floats"] = np.random.normal(size=D)
        adata.var["strings"] = np.random.choice(["abc", "def"], size=D)
        adata.var["categories"] = adata.var["strings"].astype("category")

        adata.write(filename)

        ash = AnnDataShadow(filename)
        assert adata.var.shape == ash.var.shape

        assert ash.var.strings.equals(adata.var.strings)
        assert ash.var.categories.equals(adata.var.categories)

        ash.close()

    def test_anndata_varm(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_varm.h5ad")

        for i in range(2, 10):
            adata.varm["loadings"] = np.random.normal(size=(D, 2))
            adata.write(filename)

            ash = AnnDataShadow(filename)

            assert "loadings" in ash.varm
            assert adata.varm["loadings"].shape == ash.varm["loadings"].shape

            ash.close()

    def test_anndata_uns(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_uns.h5ad")

        adata.uns["logical"] = np.random.choice([True, False])
        adata.uns["integer"] = 1
        adata.uns["float"] = 0.1
        adata.uns["string"] = "abc"
        adata.uns["dict"] = {"a": 1, "b": 2}

        adata.write(filename)

        ash = AnnDataShadow(filename)

        assert adata.uns["string"] == ash.uns["string"]
        assert adata.uns["dict"] == ash.uns["dict"]

        ash.close()


@pytest.mark.usefixtures("filepath_h5ad")
class TestViewsAnnData:
    def test_single_view_range(self, adata, filepath_h5ad):
        filename = filepath_h5ad
        adata.write(filename)

        np.random.seed(42)
        i = np.random.choice(N, 1)[0]
        j = np.random.choice(D, 1)[0]

        ash = AnnDataShadow(filename)

        view = adata[:i, :j]
        ash_view = ash[:i, :j]

        assert ash_view.shape == view.shape
        assert ash_view.shape == (i, j)
        assert ash_view.X.shape == (i, j)

        ash.close()

    def test_nested_views(self, adata, filepath_h5ad):
        filename = filepath_h5ad
        adata.write(filename)

        np.random.seed(42)
        i = np.random.choice(N, 1)[0]
        j = np.random.choice(D, 1)[0]
        ii = np.random.choice(i, 1)[0]
        jj = np.random.choice(j, 1)[0]

        ash = AnnDataShadow(filename)

        view = adata[:i, :j]
        view = view[:ii, :jj]
        ash_view = ash[:i, :j]
        ash_view = ash_view[:ii, :jj]

        assert ash_view.shape == view.shape
        assert ash_view.shape == (ii, jj)
        assert ash_view.X.shape == (ii, jj)

        assert ash_view.obs_names.equals(view.obs_names)
        assert ash_view.var_names.equals(view.var_names)

        ash.close()


@pytest.mark.usefixtures("filepath_h5mu")
@pytest.mark.usefixtures("filepath_mudata_zarr")
class TestMuData:
    def test_mudata_simple(self, mdata, filepath_h5mu):
        filename = filepath_h5mu
        mdata.write(filename)

        msh = MuDataShadow(filename)

        assert mdata.shape == msh.shape

        msh.close()

    def test_anndata_inside_mudata(self, mdata, filepath_h5mu):
        filename = filepath_h5mu
        mdata.write(filename)

        mod_x = Path(filename) / "mod" / "x"
        mod_y = Path(filename) / "mod" / "y"

        ash_x = AnnDataShadow(mod_x)
        ash_y = AnnDataShadow(mod_y)

        assert ash_x.shape == mdata["x"].shape
        assert ash_y.shape == mdata["y"].shape

        ash_x.close()
        ash_y.close()
