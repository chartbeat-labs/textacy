"""
:mod:`textacy.cache`: Functionality for caching language data and other NLP resources.
Loading data from disk can be slow; let's just do it once and forget about it. :)
"""
import inspect
import os
import sys

from cachetools import LRUCache


def _get_size(obj, seen=None):
    """
    Recursively find the actual size of an object, in bytes.

    Taken as-is (with tweaked function name) from https://github.com/bosswissam/pysize.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += _get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((_get_size(v, seen) for v in obj.values()))
        size += sum((_get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((_get_size(i, seen) for i in obj))
    return size


LRU_CACHE = LRUCache(
    int(os.environ.get("TEXTACY_MAX_CACHE_SIZE", 2147483648)), getsizeof=_get_size
)
""":class:`cachetools.LRUCache`: Least Recently Used (LRU) cache for loaded data.

The max cache size may be set by the `TEXTACY_MAX_CACHE_SIZE` environment variable,
where the value must be an integer (in bytes). Otherwise, the max size is 2GB.
"""


def clear():
    """Clear textacy's cache of loaded data."""
    global LRU_CACHE
    LRU_CACHE.clear()
