class Resource:
    """
    Base class for textacy resources.

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
        return "Resource('{}')".format(self.name)

    @property
    def info(self):
        info = {"name": self.name}
        info.update(self.meta)
        return info

    def download(self):
        raise NotImplementedError()
