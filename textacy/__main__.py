from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import sys
from pprint import pprint

from . import compat
from . import datasets
from . import lexicon_methods

# let's cheat and add a handler to the datasets logger
# whose messages we'll send to stdout
LOGGER = logging.getLogger("textacy.datasets")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
ch.setLevel(logging.INFO)
LOGGER.addHandler(ch)
LOGGER.setLevel(logging.INFO)

DATASET_NAME_TO_CLASS = {
    "capitol_words": datasets.CapitolWords,
    "imdb": datasets.IMDB,
    "oxford_text_archive": datasets.OxfordTextArchive,
    "reddit_comments": datasets.RedditComments,
    "supreme_court": datasets.SupremeCourt,
    "wikinews": datasets.Wikinews,
    "wikipedia": datasets.Wikipedia,
    "depechemood": lexicon_methods.download_depechemood,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="command line interface for textacy",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # the "download" command
    parser_download = subparsers.add_parser(
        "download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="download datasets and such",
    )
    parser_download.add_argument(
        "dataset_name",
        type=str,
        choices=list(DATASET_NAME_TO_CLASS.keys()),
        help="name of dataset to download",
    )
    parser_download.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="path on disk where dataaset will be saved on disk",
    )
    parser_download.add_argument(
        "--date_range",
        nargs=2,
        type=str,
        required=False,
        help='if `dataset_name` is "reddit_comments", the [start, end) range '
        "of dates for which comments files will be downloaded, where each "
        "item is a string formatted as YYYY-MM or YYYY-MM-DD",
    )
    parser_download.add_argument(
        "--lang",
        type=str,
        required=False,
        help='if `dataset_name` is "wikipedia" or "wikinews", language of '
        "the database dump to download",
    )
    parser_download.add_argument(
        "--version",
        type=str,
        required=False,
        help='if `dataset_name` is "wikipedia" or "wikinews", version of '
        "the database dump to download",
    )
    parser_download.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="if specified, force a download of `dataset_name` to `data_dir`, "
        "whether or not that dataset already exists in this directory",
    )
    # the "info" command
    parser_info = subparsers.add_parser(
        "info",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="get basic information about datasets and such",
    )
    parser_info.add_argument(
        "dataset_name",
        type=str,
        nargs="?",
        choices=list(DATASET_NAME_TO_CLASS.keys()),
        help="name of dataset to get basic information about",
    )

    args = vars(parser.parse_args())

    if args["subcommand"] == "download":
        # do we have a `Dataset`?
        if hasattr(DATASET_NAME_TO_CLASS[args["dataset_name"]], "download"):
            # initialize dataset
            kwargs_init = {
                key: args[key]
                for key in ["data_dir", "lang", "version"]
                if args.get(key) is not None
            }
            dataset = DATASET_NAME_TO_CLASS[args["dataset_name"]](**kwargs_init)
            # download data using the class method
            kwargs_dl = {
                key: args[key]
                for key in ["date_range", "force"]
                if args.get(key) is not None
            }
            dataset.download(**kwargs_dl)
        # if not, let's assume it's a function that takes similar kwargs
        else:
            kwargs_func = {
                key: args[key]
                for key in ["data_dir", "force"]
                if args.get(key) is not None
            }
            DATASET_NAME_TO_CLASS[args["dataset_name"]](**kwargs_func)

    if args["subcommand"] == "info":
        if args.get("dataset_name") is None:
            pprint(compat.get_config())
        # do we have a `Dataset`?
        elif hasattr(DATASET_NAME_TO_CLASS[args["dataset_name"]], "info"):
            kwargs_init = {
                key: args[key] for key in ["data_dir"] if args.get(key) is not None
            }
            dataset = DATASET_NAME_TO_CLASS[args["dataset_name"]](**kwargs_init)
            pprint(dataset.info)
        else:
            LOGGER.info("no info available for dataset %s", args["dataset_name"])

# finally, remove the handler that we snuck in above
LOGGER.removeHandler(ch)
