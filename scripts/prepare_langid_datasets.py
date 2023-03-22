import argparse
import collections
import json
import logging
import operator
import pathlib
import random
import statistics
from typing import Optional

import sklearn.model_selection
import spacy
from spacy.tokens import Doc, DocBin
from spacy.util import registry
from toolz import itertoolz

import textacy.datasets
import textacy.lang_id._datasets  # oof, naming
import textacy.lang_id.code
import textacy.preprocessing


logging.basicConfig(level=logging.INFO)


def main():
    args = add_and_parse_args()
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_agg_data(
        args.src_root_dir, args.min_text_len, args.min_obs, args.seed, args.force
    )
    # HACK: let's make sure there aren't any URLs in our training data
    # since it seems like a bunch of characters that would confuse the model
    data = [(textacy.preprocessing.replace.urls(text, ""), lang) for text, lang in data]
    summarize_data("agg", data)

    train_data, test_data = sklearn.model_selection.train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=[lang for _, lang in data],
    )
    print(f"training data: {len(train_data)}\ntest_data: {len(test_data)}")

    nlp = spacy.blank("xx")
    if args.tokenizer:
        tokenizer_func = registry.tokenizers.get(args.tokenizer)
        nlp.tokenizer = tokenizer_func(1000, True)(nlp)

    print("converting train records to docs ...")
    train_docbin = DocBin(docs=(convert_record(nlp, record) for record in train_data))
    if args.save_dir:
        train_docbin.to_disk(args.save_dir / "train.spacy")

    print("saving train labels to disk ...")
    labels = sorted(set(lang for _, lang in train_data))
    if args.save_dir:
        with args.save_dir.joinpath("labels.json").open("w") as f:
            json.dump(labels, f)

    print("converting test records to docs ...")
    test_docbin = DocBin(docs=(convert_record(nlp, record) for record in test_data))
    if args.save_dir:
        test_docbin.to_disk(args.save_dir / "test.spacy")


def add_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src-root-dir",
        type=pathlib.Path,
        required=True,
        help="path to root directory under which source datasets are saved",
    )
    parser.add_argument(
        "--save-dir",
        type=pathlib.Path,
        required=False,
        help="path to directory under which target artifacts will be saved",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default=None,
        choices=["textacy.char_tokenizer.v1"],
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=20,
        help="minimum number of alphanumeric characters in a text "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=300,
        help="minimum number of observations -- (text, lang) pairs -- in a language "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="fraction of data observations to set aside for the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed number used to make random operations deterministic, for reproducibility",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="if specified, force downloads of all datasets, "
        "even if they already exist on disk under ``src_root_dir``",
    )
    return parser.parse_args()


def load_and_agg_data(
    src_root_dir: pathlib.Path,
    min_text_len: int,
    min_obs: int,
    seed: Optional[int],
    force: bool,
) -> list[tuple[str, str]]:
    """Download, load, and aggregate datasets."""
    iso_lang_resource = textacy.lang_id._datasets.IsoLangResource(
        src_root_dir.joinpath("iso-639")
    )
    iso_lang_resource.download(force=force)
    iso_lang_map = iso_lang_resource.load(exclude={"sh"})  # TODO: why exclude sh?
    valid_langs = set(iso_lang_map.values())

    udhr = textacy.datasets.UDHR(src_root_dir.joinpath("udhr"))
    udhr.download(force=force)
    udhr_data = [
        (snippet, meta["lang"])
        for text, meta in udhr.records()
        for snippet in text.split("\n")
        if meta["lang"] in valid_langs
        and itertoolz.count(char for char in snippet if char.isalnum()) >= min_text_len
    ]
    random.shuffle(udhr_data)

    dslcc = textacy.lang_id._datasets.DSLCCDataset(src_root_dir.joinpath("dslcc"))
    dslcc.download(force=force)
    dslcc_data = dslcc.load(valid_langs, min_len=min_text_len)

    wili = textacy.lang_id._datasets.Wili2018Dataset(src_root_dir.joinpath("wili"))
    wili.download(force=force)
    wili_data = wili.load(iso_lang_map, min_len=min_text_len)
    random.shuffle(udhr_data)

    tatoeba = textacy.lang_id._datasets.TatoebaDataset(src_root_dir.joinpath("tatoeba"))
    tatoeba.download(force=force)
    tatoeba_data = tatoeba.load(iso_lang_map, min_len=min_text_len)

    ud = textacy.lang_id._datasets.UDDataset(src_root_dir.joinpath("ud"))
    ud.download(force=force)
    ud_data = ud.load(valid_langs, min_len=min_text_len)

    # aggregate and sample datasets
    agg_data = (
        udhr_data
        + wili_data
        + get_random_sample(tatoeba_data, 200000, stratify=True, random_state=seed)
        + get_random_sample(ud_data, 200000, stratify=True, random_state=seed)
        # add additional examples for hard-to-distinguish language groups
        + get_random_sample(dslcc_data, 50000, stratify=True, random_state=seed)
        # add some extra english examples, since there's apparently a fair amount
        # of english sprinkled throughout other languages, causing meh performance
        + get_random_sample(
            [item for item in tatoeba_data if item[1] == "en"],
            10000,
            stratify=False,
            random_state=seed,
        )
    )

    # agg_data = get_random_sample(
    #     tatoeba_data, 1_000_000, stratify=True, random_state=seed
    # )

    agg_data = filter_data_by_lang_count(agg_data, min_obs)

    return agg_data


def get_random_sample(
    seq, n: int, stratify: bool = True, random_state: Optional[int] = None
) -> list:
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
    data: list[tuple[str, str]], min_obs: int
) -> list[tuple[str, str]]:
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


def summarize_data(name: str, data: list[tuple[str, str]]):
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


def convert_record(nlp: spacy.language.Language, record: tuple[str, str]) -> Doc:
    """Convert a record from the tsv into a spaCy Doc object."""
    doc = nlp.make_doc(record[0])
    doc.cats = {record[1]: 1.0}
    # # All categories other than the true ones get value 0
    # doc.cats = {category: 0 for category in categories}
    # # True labels get value 1
    # for label in record["labels"]:
    #     doc.cats[categories[label]] = 1
    return doc


if __name__ == "__main__":
    main()
