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

- [x] AnnDataShadow and MuDataShadow for Parquet-based serialization ([pqdata](https://github.com/gtca/pqdata)).

- [x] Data shadows for `.pqdata` and `.zarr` files on S3 storage.


### Shadow objects

Briefly, shadow objects simply work like this:

```py
from shadows import *
ash = AnnDataShadow("pbmc3k.h5ad")
msh = MuDataShadow("pbmc5k_citeseq.h5mu")
```

All the various features are showcased in the following tutorials:

- [Getting started with shadow objects](/docs/examples/shadow-objects.ipynb)

- [Advanced features of shadow objects](/docs/examples/shadows-features.ipynb)

[//]: # (numfocus-fiscal-sponsor-attribution)

shadows is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

