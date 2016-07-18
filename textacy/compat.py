import bz2
import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from itertools import izip as zip
    from backports import lzma

    bytes_type = str
    unicode_type = unicode
    string_types = (str, unicode)
    str = unicode  # TODO: don't do this
    bzip_open = bz2.BZ2File  # drop this

    def unicode_to_bytes(s, encoding=None, errors=None):
        return s

    def bytes_to_unicode(b, encoding=None, errors=None):
        return b

else:
    import lzma

    zip = zip
    bytes_type = bytes
    unicode_type = str
    string_types = (bytes, str)
    str = str  # TODO: don't do this
    bzip_open = bz2.open  # drop this

    def unicode_to_bytes(s, encoding='utf8', errors='strict'):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding='utf8', errors='strict'):
        return b.decode(encoding=encoding, errors=errors)
