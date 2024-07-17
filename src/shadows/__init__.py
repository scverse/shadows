from .anndatashadow import AnnDataShadow
from .datashadow import DataShadow
from .mudatashadow import MuDataShadow

try:  # See https://github.com/maresb/hatch-vcs-footgun-example
    from setuptools_scm import get_version

    __version__ = get_version(root="../..", relative_to=__file__)
except (ImportError, LookupError):
    try:
        from ._version import __version__
    except ModuleNotFoundError:
        raise RuntimeError("pqdata is not correctly installed. Please install it, e.g. with pip.")

__all__ = ["DataShadow", "AnnDataShadow", "MuDataShadow", "__version__"]
