from anndata._io.specs import read_elem as ad_read_elem

try:
    from pqdata.core import Array as PqArray
    from pqdata.core import Group as PqGroup
except ImportError:

    class PqArray:
        @staticmethod
        def __repr__():
            return "mock pqdata.core.Array"

    class PqGroup:
        @staticmethod
        def __repr__():
            return "mock pqdata.core.Group"


def read_elem(*args, **kwargs):
    if "_format" in kwargs:
        format = kwargs.pop("_format")
        if format == "parquet":
            from pqdata.core import read_elem as pq_read_elem

            return pq_read_elem(*args, **kwargs)
    else:
        return ad_read_elem(*args, **kwargs)
