from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from pprint import pprint

import textacy.datasets

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

    # download command
    parser_download = subparsers.add_parser(
        'download', help='download datasets and such')
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

    parser_info = subparsers.add_parser(
        'info', help='get basic information about datasets and such')
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
        download_args = ['date_range', 'lang', 'version', 'force']
        dataset.download(
            **{key: args[key] for key in download_args if args.get(key) is not None})

    if args['subcommand'] == 'info':
        dataset = DATASET_NAME_TO_CLASS[args['dataset_name']]()
        pprint(dataset.info)