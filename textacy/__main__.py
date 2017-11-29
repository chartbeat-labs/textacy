from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from pprint import pprint
import sys

import textacy.datasets

# let's cheat and add a handler to the datasets logger
# whose messages we'll send to stdout
LOGGER = logging.getLogger('textacy.datasets')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
ch.setLevel(logging.INFO)
LOGGER.addHandler(ch)
LOGGER.setLevel(logging.INFO)

DATASET_NAME_TO_CLASS = {
    'capitol_words': textacy.datasets.CapitolWords,
    'oxford_text_archive': textacy.datasets.OxfordTextArchive,
    'reddit_comments': textacy.datasets.RedditComments,
    'supreme_court': textacy.datasets.SupremeCourt,
    'wikipedia': textacy.datasets.Wikipedia,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='subcommand')

    # the "download" command
    parser_download = subparsers.add_parser(
        'download', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='download datasets and such')
    parser_download.add_argument(
        'dataset_name', type=str, choices=list(DATASET_NAME_TO_CLASS.keys()),
        help='name of dataset to download')
    parser_download.add_argument(
        '--data_dir', type=str, required=False,
        help='path on disk where dataaset will be saved on disk')
    parser_download.add_argument(
        '--date_range', nargs=2, type=str, required=False,
        help='if `dataset_name` is "reddit_comments", the [start, end) range '
             'of dates for which comments files will be downloaded, where each '
             'item is a string formatted as YYYY-MM or YYYY-MM-DD')
    parser_download.add_argument(
        '--lang', type=str, required=False,
        help='if `dataset_name` is "wikipedia", language of wikipedia '
             'database dump to download')
    parser_download.add_argument(
        '--version', type=str, required=False,
        help='if `dataset_name` is "wikipedia", version of wikipedia '
             'database dump to download')
    parser_download.add_argument(
        '--force', default=False, action='store_true',
        help='if specified, force a download of `dataset_name` to `data_dir`, '
             'whether or not that dataset already exists in this directory')
    # the "info" command
    parser_info = subparsers.add_parser(
        'info', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='get basic information about datasets and such')
    parser_info.add_argument(
        'dataset_name', type=str, choices=list(DATASET_NAME_TO_CLASS.keys()),
        help='name of dataset to get basic information about')

    args = vars(parser.parse_args())

    if args['subcommand'] == 'download':
        # initialize dataset
        if args.get('data_dir'):
            dataset = DATASET_NAME_TO_CLASS[args['dataset_name']](data_dir=args['data_dir'])
        else:
            dataset = DATASET_NAME_TO_CLASS[args['dataset_name']]()
        # download data using the class method
        kwargs = {
            key: args[key]
            for key in ['date_range', 'lang', 'version', 'force']
            if args.get(key) is not None}
        dataset.download(**kwargs)

    if args['subcommand'] == 'info':
        dataset = DATASET_NAME_TO_CLASS[args['dataset_name']]()
        pprint(dataset.info)

# finally, remove the handler that we snuck in above
LOGGER.removeHandler(ch)
