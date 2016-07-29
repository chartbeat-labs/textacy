import os

from textacy import __data_dir__
from textacy.compat import PY2, string_types
from textacy.fileio import read_json_lines


class CapitolWords(object):
    """
    TODO
    """

    speaker_names = {
        'Barack Obama', 'Bernie Sanders', 'Hillary Clinton', 'Jim Webb', 'Joe Biden',
        'John Kasich', 'Joseph Biden', 'Lincoln Chafee', 'Lindsey Graham', 'Marco Rubio',
        'Mike Pence', 'Rand Paul', 'Rick Santorum', 'Ted Cruz'}
    speaker_parties = {'R', 'D', 'I'}
    chambers = {'Extensions', 'House', 'Senate'}
    congresses = {104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114}

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = __data_dir__
        if PY2:
            self.filepath = os.path.join(data_dir, 'capitol-words.json')
        else:
            self.filepath = os.path.join(data_dir, 'capitol-words.json.gz')

    def __iter__(self):
        for line in read_json_lines(self.filepath, mode='rt'):
            yield line

    def texts(self, speaker_name=None, speaker_party=None, chamber=None,
              congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over texts in the CapitolWords corpus, optionally filtering by
        a variety of metadata and/or text length.

        Args:
            speaker_name (str or set[str])
            speaker_party (str or set[str])
            chamber (str or set[str])
            congress (int or set[int])
            date_range (list[str] or tuple[str])
            min_len (int)
            limit (int)

        Yields:
            str

        Raises:
            ValueError
        """
        # prepare filters
        if speaker_name:
            if isinstance(speaker_name, string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError()
        if speaker_party:
            if isinstance(speaker_party, string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError()
        if chamber:
            if isinstance(chamber, string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError()
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError()
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                raise ValueError()
            if not len(date_range) == 2:
                raise ValueError()

        n = 0
        for doc in self:

            if speaker_name and doc['speaker_name'] not in speaker_name:
                continue
            if speaker_party and doc['speaker_party'] not in speaker_party:
                continue
            if chamber and doc['chamber'] not in chamber:
                continue
            if congress and doc['congress'] not in congress:
                continue
            if date_range and not date_range[0] <= doc['date'] <= date_range[1]:
                continue
            if min_len and len(doc['text']) < min_len:
                continue

            yield doc['text']

            n += 1
            if n == limit:
                break

    def docs(self, speaker_name=None, speaker_party=None, chamber=None,
             congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over documents (including text and metadata) in the CapitolWords
        corpus, optionally filtering by a variety of metadata and/or text length.

        Args:
            speaker_name (str or set[str])
            speaker_party (str or set[str])
            chamber (str or set[str])
            congress (int or set[int])
            date_range (list[str] or tuple[str])
            min_len (int)
            limit (int)

        Yields:
            str

        Raises:
            ValueError
        """
        # prepare filters
        if speaker_name:
            if isinstance(speaker_name, string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError()
        if speaker_party:
            if isinstance(speaker_party, string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError()
        if chamber:
            if isinstance(chamber, string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError()
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError()
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                raise ValueError()
            if not len(date_range) == 2:
                raise ValueError()

        n = 0
        for doc in self:

            if speaker_name and doc['speaker_name'] not in speaker_name:
                continue
            if speaker_party and doc['speaker_party'] not in speaker_party:
                continue
            if chamber and doc['chamber'] not in chamber:
                continue
            if congress and doc['congress'] not in congress:
                continue
            if date_range and not date_range[0] <= doc['date'] <= date_range[1]:
                continue
            if min_len and len(doc['text']) < min_len:
                continue

            yield doc

            n += 1
            if n == limit:
                break
