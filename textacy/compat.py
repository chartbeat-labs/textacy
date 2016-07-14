import bz2
import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from itertools import izip
    zip = izip
    bytes_type = str
    unicode_type = unicode
    str = unicode  # TODO: don't do this
    bzip_open = bz2.BZ2File
else:
    zip = zip
    bytes_type = bytes
    unicode_type = str
    str = str  # TODO: don't do this
    bzip_open = bz2.open
