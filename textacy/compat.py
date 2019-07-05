from __future__ import unicode_literals

import sys

PY2 = sys.version_info.major == 2
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_osx = sys.platform == "darwin"
is_narrow_unicode = sys.maxunicode < 0x10ffff

if PY2:
    import cPickle as pickle
    from backports import csv
    from collections import Iterable
    from itertools import izip as zip_
    from urllib import unquote_plus as url_unquote_plus
    from urlparse import urljoin, urlparse

    from numpy import mean as mean_
    from numpy import median as median_
    from numpy import std as stdev_

    from math import log as _log

    def log2_(x):
        """Return the base-2 logarithm of x."""
        return _log(x, 2)

    reduce_ = reduce
    range_ = xrange

    unicode_ = unicode
    string_types = (bytes, unicode)
    int_types = (int, long)
    chr_ = unichr

else:
    import csv
    import pickle
    from builtins import zip as zip_
    from collections.abc import Iterable
    from functools import reduce as reduce_
    from math import log2 as log2_
    from statistics import mean as mean_
    from statistics import median as median_
    from statistics import stdev as stdev_
    from urllib.parse import unquote_plus as url_unquote_plus
    from urllib.parse import urljoin, urlparse

    range_ = range

    unicode_ = str
    string_types = (bytes, str)
    int_types = (int,)
    chr_ = chr


def to_bytes(s, encoding="utf-8", errors="strict"):
    """Coerce ``s`` to bytes.

    Args:
        s (unicode or bytes)
        encoding (str)
        errors (str)

    Returns:
        bytes
    """
    if isinstance(s, unicode_):
        return s.encode(encoding, errors)
    elif isinstance(s, bytes):
        return s
    else:
        raise TypeError("`s` must be {}, not {}".format(string_types, type(s)))


def to_unicode(s, encoding="utf-8", errors="strict"):
    """Coerce ``s`` to unicode.

    Args:
        s (unicode or bytes)
        encoding (str)
        errors (str)

    Returns:
        unicode
    """
    if isinstance(s, bytes):
        return s.decode(encoding, errors)
    elif isinstance(s, unicode_):
        return s
    else:
        raise TypeError("`s` must be {}, not {}".format(string_types, type(s)))
