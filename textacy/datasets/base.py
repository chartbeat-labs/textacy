import logging
import os

from ..io.utils import open_sesame

LOGGER = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name, description, site_url, data_dir):
        """
        Args:
            name (str)
            description (str)
            site_url (str)
            data_dir (str)
        """
        self.name = name
        self.description = description
        self.site_url = site_url
        self.data_dir = data_dir

    def __repr__(self):
        return 'Dataset("{}", description="{}", site_url="{}", data_dir="{}")'.format(
            self.name, self.description, self.site_url, self.data_dir
        )

    @property
    def info(self):
        return {
            "name": self.name,
            "description": self.description,
            "site_url": self.site_url,
            "data_dir": self.data_dir,
        }

    def download(self):
        raise NotImplementedError()

    def texts(self):
        raise NotImplementedError()

    def records(self):
        raise NotImplementedError()

    def _parse_date_range(self, date_range):
        """
        Flexibly parse date range args, where ``date_range`` is length-2 list or
        tuple for which null values will be automatically set equal to the min
        or max valid dates, if available as class attributes.
        """
        if not isinstance(date_range, (list, tuple)):
            raise ValueError(
                "`date_range` must be a list or tuple, not {}".format(type(date_range))
            )
        if len(date_range) != 2:
            raise ValueError("`date_range` must have exactly two items: start and end")
        if not date_range[0]:
            try:
                date_range = (self.min_date, date_range[1])
            except AttributeError:
                raise ValueError("`date_range` minimum must be specified")
        if not date_range[1]:
            try:
                date_range = (date_range[0], self.max_date)
            except AttributeError:
                raise ValueError("`date_range` maximum must be specified")
        # check date range bounds
        if date_range[0] < self.min_date:
            LOGGER.warning(
                "start of date_range %s < minimum valid date %s; clipping range "
                "accordingly",
                date_range[0],
                self.min_date,
            )
            date_range[0] = self.min_date
        if date_range[1] > self.max_date:
            LOGGER.warning(
                "end of date_range %s > maximum valid date %s; clipping range "
                "accordingly",
                date_range[1],
                self.max_date,
            )
            date_range[1] = self.max_date
        return tuple(date_range)
