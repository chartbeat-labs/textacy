from typing import Optional


class Dataset:
    """
    Base class for textacy datasets.

    Args:
        name (str)
        meta (dict)

    Attributes:
        name (str)
        meta (dict)
    """

    def __init__(self, name: str, meta: Optional[dict] = None):
        self.name = name
        self.meta = meta or {}

    def __repr__(self):
        return f"Dataset('{self.name}')"

    @property
    def info(self) -> dict[str, str]:
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
