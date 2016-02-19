import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from itertools import izip
    zip = izip
    str = unicode
else:
    zip = zip
    str = str
