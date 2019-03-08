from __future__ import print_function

import sys

is_python2 = int(sys.version[0]) == 2
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")
is_osx = sys.platform == "darwin"

if is_python2:
    import cPickle as pickle
    from backports import csv
    from itertools import izip as zip_
    from urlparse import urljoin

    range_ = xrange

    bytes_ = str
    unicode_ = unicode
    string_types = (str, unicode)
    int_types = (int, long)
    chr_ = unichr

    def unicode_to_bytes(s, encoding="utf8", errors="strict"):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding="utf8", errors="strict"):
        return unicode_(b, encoding=encoding, errors=errors)


else:
    import csv
    import pickle
    from builtins import zip as zip_
    from urllib.parse import urljoin

    range_ = range

    bytes_ = bytes
    unicode_ = str
    string_types = (bytes, str)
    int_types = (int,)
    chr_ = chr

    def unicode_to_bytes(s, encoding="utf8", errors="strict"):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding="utf8", errors="strict"):
        return b.decode(encoding=encoding, errors=errors)
