import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from backports import csv
    from itertools import izip as zip

    bytes_type = str
    unicode_type = unicode
    string_types = (str, unicode)

    def unicode_to_bytes(s, encoding='utf8', errors='strict'):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding='utf8', errors='strict'):
        return unicode_type(b, encoding=encoding, errors=errors)

else:
    import csv

    zip = zip
    bytes_type = bytes
    unicode_type = str
    string_types = (bytes, str)

    def unicode_to_bytes(s, encoding='utf8', errors='strict'):
        return s.encode(encoding=encoding, errors=errors)

    def bytes_to_unicode(b, encoding='utf8', errors='strict'):
        return b.decode(encoding=encoding, errors=errors)
