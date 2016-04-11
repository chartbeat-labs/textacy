import bz2
import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from itertools import izip
    zip = izip
    str = unicode
    bzip_open = bz2.BZ2File
else:
    zip = zip
    str = str
    bzip_open = bz2.open
