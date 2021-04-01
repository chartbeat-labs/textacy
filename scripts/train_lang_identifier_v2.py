from __future__ import annotations

import argparse
import collections
import logging
import operator
import pathlib
import random
import statistics
from typing import List, Tuple

import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import thinc.api
from cytoolz import itertoolz

import textacy
import textacy.datasets
import textacy.lang_id_
import textacy.lang_id_._datasets  # oof, naming
import textacy.lang_id_._training
import textacy.preprocessing

logging.basicConfig(level=logging.INFO)


def main():
    args = add_and_parse_args()

    lang_identifier = textacy.lang_id_.LangIdentifier(
        textacy.lang_id_.models.LangIdentifierModelV2(),
        args.root_dirpath.joinpath("lang_identifier"),
        version=args.version,
    )

    data = load_and_agg_data(args.root_dirpath, args.force, args.min_len, args.min_obs)
    # HACK: let's make sure there aren't any URLs in our training data
    # since it seems like a bunch of characters that would confuse the model
    data = [
        (textacy.preprocessing.replace.urls(text, ""), lang)
        for text, lang in data
    ]
    summarize_data("agg", data)

    train_data, test_data = sklearn.model_selection.train_test_split(
        data, test_size=0.2, random_state=42,
        stratify=[lang for _, lang in data]
    )
    print(f"training data: {len(train_data)}\ntest_data: {len(test_data)}")

    batch_size = thinc.api.compounding(2.0, 64.0, 1.001)
    # NOTE: training appears to be VERY sensitive to learning rate
    # these values have been manually fine-tuned ...
    # learn_rate = thinc.api.cyclic_triangular(min_lr=0.0005, max_lr=0.005, period=2500)
    learn_rate = textacy.lang_id_._training.decaying_cyclic_triangular(
        thinc.api.decaying(1.0, decay=1e-4),
        thinc.api.cyclic_triangular(min_lr=0.0005, max_lr=0.005, period=2500),
        min_lr=0.00025,
    )
    model = textacy.lang_id_._training.train_model(
        lang_identifier.model,
        train=train_data,
        test=test_data,
        n_iter=args.n_iter,
        batch_size=batch_size,
        learn_rate=learn_rate,
    )
    lang_identifier.model = model
    if args.save:
        lang_identifier.save_model()


def add_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_dirpath",
        type=pathlib.Path,
        required=True,
        help="path to root directory under which datasets and models are saved",
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="semantic version number to assign to trained model, e.g. '2.0'",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=20,
        help="minimum number of alphanumeric characters in a text "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--min_obs",
        type=int,
        default=300,
        help="minimum number of observations -- (text, lang) pairs -- in a language "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="number of epochs to train model",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="if specified, force downloads of all datasets, "
        "even if they already exist on disk under ``root_dirpath``",
    )
    return parser.parse_args()


def load_and_agg_data(
    root_dirpath: pathlib.Path, force: bool, min_len: int, min_obs: int
) -> List[Tuple[str, str]]:
    """Download, load, and aggregate datasets."""
    iso_lang_resource = textacy.lang_id_._datasets.IsoLangResource(
        root_dirpath.joinpath("iso-639")
    )
    iso_lang_resource.download(force=force)
    iso_lang_map = iso_lang_resource.load(exclude={"sh"})  # TODO: why exclude sh?
    valid_langs = set(iso_lang_map.values())

    udhr = textacy.datasets.UDHR(root_dirpath.joinpath("udhr"))
    udhr.download(force=force)
    udhr_data = [
        (snippet, meta["lang"])
        for text, meta in udhr.records()
        for snippet in text.split("\n")
        if meta["lang"] in valid_langs and
        itertoolz.count(char for char in snippet if char.isalnum()) >= min_len
    ]

    dslcc = textacy.lang_id_._datasets.DSLCCDataset(root_dirpath.joinpath("dslcc"))
    dslcc.download(force=force)
    dslcc_data = dslcc.load(valid_langs, min_len=min_len)

    wili = textacy.lang_id_._datasets.Wili2018Dataset(root_dirpath.joinpath("wili"))
    wili.download(force=force)
    wili_data = wili.load(iso_lang_map, min_len=min_len)

    tatoeba = textacy.lang_id_._datasets.TatoebaDataset(root_dirpath.joinpath("tatoeba"))
    tatoeba.download(force=force)
    tatoeba_data = tatoeba.load(iso_lang_map, min_len=min_len)

    # aggregate and sample datasets
    agg_data = (
        udhr_data
        + wili_data
        + get_random_sample(tatoeba_data, 20000, stratify=True, random_state=42)
        # add additional examples for hard-to-distinguish language groups
        + get_random_sample(dslcc_data, 5000, stratify=True, random_state=42)
        # add some extra english examples, since there's apparently a fair amount
        # of english sprinkled throughout other languages, causing meh performance
        + get_random_sample(
            [item for item in tatoeba_data if item[1] == "en"],
            1000,
            stratify=False,
            random_state=42,
        )
    )
    agg_data = filter_data_by_lang_count(agg_data, min_obs)
    return agg_data


def summarize_data(name: str, data: List[Tuple[str, str]]):
    print(f"\n{name.upper()}")
    print(f"# observations: {len(data)}\n{data[:3]} ...")
    print(
        f"min text len: {min(len(text) for text, _ in data)}\n"
        f"mean text len: {statistics.mean(len(text) for text, _ in data)}\n"
        f"stdev text len: {statistics.stdev(len(text) for text, _ in data)}\n"
        f"max text len: {max(len(text) for text, _ in data)}"
    )
    lang_counts = collections.Counter(lang for _, lang in data)
    top_counts = "; ".join(
        f"{lang}: {count}" for lang, count in lang_counts.most_common(15)
    )
    bot_counts = "; ".join(
        f"{lang}: {count}"
        for lang, count in sorted(
            lang_counts.items(), key=operator.itemgetter(1), reverse=True
        )[-15:]
    )
    print(f"# unique chars: {len({char for text, _ in data for char in text})}")
    print(f"# unique languages: {len(lang_counts)}\n{top_counts} ... \n{bot_counts}")


def get_random_sample(seq, n, stratify=True, random_state=None):
    """
    Args:
        seq (Sequence)
        n (int)
        stratify (bool)
        random_state (int)

    Returns:
        list
    """
    random.seed(a=random_state)
    if stratify is True:
        grped = itertoolz.groupby(operator.itemgetter(1), seq)
        n_per_grp = max(int(round(n / len(grped))), 1)
        sample = list(
            itertoolz.concat(
                random.sample(examples, min(len(examples), n_per_grp))
                for examples in grped.values()
            )
        )
        random.shuffle(sample)
        return sample[:n]
    else:
        return random.sample(seq, min(len(seq), n))


def filter_data_by_lang_count(
    data: List[Tuple[str, str]],
    min_obs: int,
) -> List[Tuple[str, str]]:
    """
    Args:
        data
        min_obs
    """
    valid_langs = {
        lang
        for lang, count in collections.Counter(lang for _, lang in data).most_common()
        if count >= min_obs
    }
    return [text_lang for text_lang in data if text_lang[1] in valid_langs]


if __name__ == "__main__":
    main()
