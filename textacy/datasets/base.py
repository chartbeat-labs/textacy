import os

from textacy.fileio import open_sesame


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
            self.name, self.description, self.site_url, self.data_dir)

    @property
    def info(self):
        return {'name': self.name,
                'description': self.description,
                'site_url': self.site_url,
                'data_dir': self.data_dir}

    def download(self, outdir=None):
        raise NotImplementedError()

    def load(self, indir=None):
        raise NotImplementedError()
