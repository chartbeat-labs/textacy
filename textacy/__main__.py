from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

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
    parser.add_argument(
        'dataset_name', type=str, choices=list(DATASET_NAME_TO_CLASS.keys())
    )
    parser.add_argument(
        '--data_dir', type=str, required=False,
    )
    parser.add_argument(
        '--date_range', type=str, required=False,
    )
    parser.add_argument(
        '--lang', type=str, required=False,
    )
    parser.add_argument(
        '--version', type=str, required=False,
    )
    parser.add_argument(
        '--force', default=False, action='store_true',
    )
    args = vars(parser.parse_args())

    # initialize dataset
    if args.get('data_dir'):
        dataset = DATASET_NAME_TO_CLASS[args['dataset_name']](data_dir=args['data_dir'])
    else:
        dataset = DATASET_NAME_TO_CLASS[args['dataset_name']]()

    dataset.download(
        **{key: args[key] for key in ['date_range', 'lang', 'version' 'force'] if args.get(key)}
    )
