import argparse
import logging
import sys
from pprint import pprint

from . import datasets
from . import lang_id
from . import resources
from . import utils

# let's cheat and add a handler to the datasets logger
# whose messages we'll send to stdout
# TODO: figure out if this no longer works bc of non-dataset items
LOGGER = logging.getLogger("textacy.datasets")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
ch.setLevel(logging.INFO)
LOGGER.addHandler(ch)
LOGGER.setLevel(logging.INFO)

NAME_TO_CLASS = {
    "lang_identifier": lang_id.LangIdentifier,
    "capitol_words": datasets.CapitolWords,
    "imdb": datasets.IMDB,
    "oxford_text_archive": datasets.OxfordTextArchive,
    "reddit_comments": datasets.RedditComments,
    "supreme_court": datasets.SupremeCourt,
    "udhr": datasets.UDHR,
    "wikinews": datasets.Wikinews,
    "wikipedia": datasets.Wikipedia,
    "concept_net": resources.ConceptNet,
    "depeche_mood": resources.DepecheMood,
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
        help="download datasets and resources",
    )
    parser_download.add_argument(
        "name",
        type=str,
        choices=list(NAME_TO_CLASS.keys()),
        help="name of dataset or resource to download",
    )
    parser_download.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="path on disk where dataaset/resource will be saved on disk",
    )
    parser_download.add_argument(
        "--date_range",
        nargs=2,
        type=str,
        required=False,
        help='if `name` is "reddit_comments", the [start, end) range '
        "of dates for which comments files will be downloaded, where each "
        "item is a string formatted as YYYY-MM or YYYY-MM-DD",
    )
    parser_download.add_argument(
        "--lang",
        type=str,
        required=False,
        help='if `name` is "wikipedia" or "wikinews", language of '
        "the database dump to download",
    )
    parser_download.add_argument(
        "--version",
        type=str,
        required=False,
        help='if `name` is "lang_identifier", "wikipedia", "wikinews", or "concept_net", '
        "version of the database to download",
    )
    parser_download.add_argument(
        "--force",
        default=False,
        action="store_true",
        help="if specified, force a download of `name` to `data_dir`, "
        "whether or not that resource already exists in this directory",
    )
    # the "info" command
    parser_info = subparsers.add_parser(
        "info",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="get basic information about datasets and resources",
    )
    parser_info.add_argument(
        "name",
        type=str,
        nargs="?",
        choices=list(NAME_TO_CLASS.keys()),
        help="name of dataset or resource to get basic information for",
    )

    args = vars(parser.parse_args())

    if args["subcommand"] == "download":
        # do we have a `Dataset` or similar resource class?
        if hasattr(NAME_TO_CLASS[args["name"]], "download"):
            # initialize resource
            kwargs_init = {
                key: args[key]
                for key in ["data_dir", "lang", "version"]
                if args.get(key) is not None
            }
            resource = NAME_TO_CLASS[args["name"]](**kwargs_init)
            # download data using the class method
            kwargs_dl = {
                key: args[key]
                for key in ["date_range", "force"]
                if args.get(key) is not None
            }
            resource.download(**kwargs_dl)
        # if not, let's assume it's a function that takes similar kwargs
        else:
            kwargs_func = {
                key: args[key]
                for key in ["data_dir", "force"]
                if args.get(key) is not None
            }
            NAME_TO_CLASS[args["name"]](**kwargs_func)

    if args["subcommand"] == "info":
        if args.get("name") is None:
            pprint(utils.get_config())
        # do we have a `Dataset` or similar resource class?
        elif hasattr(NAME_TO_CLASS[args["name"]], "info"):
            kwargs_init = {
                key: args[key] for key in ["data_dir"] if args.get(key) is not None
            }
            resource = NAME_TO_CLASS[args["name"]](**kwargs_init)
            pprint(resource.info)
        else:
            LOGGER.info("no info available for %s", args["name"])

# finally, remove the handler that we snuck in above
LOGGER.removeHandler(ch)
