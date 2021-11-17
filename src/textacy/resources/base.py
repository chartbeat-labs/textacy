from typing import Optional


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

    def __init__(self, name: str, meta: Optional[dict] = None):
        self.name = name
        self.meta = meta or {}

    def __repr__(self) -> str:
        return f"Resource('{self.name}')"

    @property
    def info(self) -> dict:
        info = {"name": self.name}
        info.update(self.meta)
        return info

    def download(self):
        raise NotImplementedError()
