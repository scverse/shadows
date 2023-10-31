<img src="./docs/img/shadows_header.jpg" data-canonical-src="./docs/img/shadows_header.svg" width="100%"/>

# Shadows

Shadows are on-disk interfaces for scverse data standards such as [AnnData](https://github.com/scverse/anndata) and [MuData](https://github.com/scverse/mudata).

It is an experimental project.

[![PyPi version](https://img.shields.io/pypi/v/shadows)](https://pypi.org/project/shadows)

## Installation

```
pip install shadows
# or
pip install git+https://github.com/scverse/shadows
```

## Features

The shadows library aims to implement the following features:

- [x] **Shadow objects**: Read-only AnnDataShadow and MuDataShadow for HDF5 files.

- [x] AnnDataShadow and MuDataShadow for Zarr files.

- [ ] AnnDataShadow and MuDataShadow for Parquet-based serialization.

- [ ] Data shadows for Laminate files on S3 storage.


### Shadow objects

[Example notebook](/docs/examples/shadow-objects.ipynb) | [More features](/docs/examples/shadow-objects-features.ipynb)

Briefly, it simply works like this:

```py
from shadows import *
ash = AnnDataShadow("pbmc3k.h5ad")
msh = MuDataShadow("pbmc5k_citeseq.h5mu")
```

