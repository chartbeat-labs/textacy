from __future__ import absolute_import, division, print_function, unicode_literals


class Dataset(object):
    """
    Base class for textacy datasets.

    Args:
        name (str)
        meta (dict)

    Attributes:
        name (str)
        meta (dict)
    """

    def __init__(self, name, meta=None):
        self.name = name
        self.meta = meta or {}

    def __repr__(self):
        return 'Dataset("{}")'.format(self.name)

    @property
    def info(self):
        info = {"name": self.name}
        info.update(self.meta)
        return info

    def __iter__(self):
        raise NotImplementedError()

    def texts(self):
        raise NotImplementedError()

    def records(self):
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()
