import pytest
from typing import Optional

from shadows import AnnDataShadow, MuDataShadow

from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from anndata import AnnData

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

    def test_anndata_varm(self, adata, filepath_h5ad):
        filename = filepath_h5ad.replace(".h5ad", "_varm.h5ad")

        for i in range(2, 10):
            adata.varm["loadings"] = np.random.normal(size=(D, 2))
            adata.write(filename)

            ash = AnnDataShadow(filename)

            assert "loadings" in ash.varm
            assert adata.varm["loadings"].shape == ash.varm["loadings"].shape

            ash.close()
