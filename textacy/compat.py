from __future__ import unicode_literals

import sys

PY2 = sys.version_info.major == 2
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_osx = sys.platform == "darwin"

if PY2:
    import cPickle as pickle
    from backports import csv
    from collections import Iterable
    from itertools import izip as zip_
    from urllib import unquote_plus as url_unquote_plus
    from urlparse import urljoin, urlparse

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
