import argparse
import collections
import logging
import operator
import pathlib
import random
import statistics
from functools import partial
from typing import Optional

import sklearn.model_selection
from toolz import itertoolz

import textacy.datasets
import textacy.io as tio
import textacy.lang_id._datasets  # oof, naming
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
    # let's also normalize the whitespace
    preproc = textacy.preprocessing.make_pipeline(
        partial(textacy.preprocessing.replace.urls, repl=""),
        textacy.preprocessing.normalize.whitespace,
        lambda x: x.replace("\n", " ").lower(),
    )
    data = ((preproc(text), lang) for text, lang in data)
    data = [item for item in data if len(item[0]) >= args.min_text_len]
    summarize_data("agg", data)

    train_data, test_data = sklearn.model_selection.train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=[lang for _, lang in data],
    )
    test_data, valid_data = sklearn.model_selection.train_test_split(
        test_data,
        test_size=0.5,
        random_state=args.seed,
        stratify=[lang for _, lang in test_data],
    )
    print(
        f"training data: {len(train_data)}\n"
        f"test_data: {len(test_data)}\n"
        f"valid_data: {len(valid_data)}"
    )

    format_and_save_data(train_data, "train", args.save_dir)
    format_and_save_data(test_data, "test", args.save_dir)
    format_and_save_data(valid_data, "valid", args.save_dir)


def format_and_save_data(
    data: list[tuple[str, str]], name: str, save_dir: Optional[pathlib.Path] = None
):
    lines = (f"__label__{lang} {text}" for text, lang in data)
    if save_dir:
        file_path = save_dir / f"{name}.txt"
        tio.text.write_text(lines, file_path, lines=True, make_dirs=True)
        print(f"saved {name} data to disk at {file_path}")


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
        "--min-text-len",
        type=int,
        default=20,
        help="minimum number of alphanumeric characters in a text "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=1_000,
        help="minimum number of observations -- (text, lang) pairs -- in a language "
        "for it to be included in the training dataset",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
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

    tatoeba = textacy.lang_id._datasets.TatoebaDataset(src_root_dir.joinpath("tatoeba"))
    tatoeba.download(force=force)
    tatoeba_data = tatoeba.load(iso_lang_map, min_len=min_text_len)

    ted2020 = textacy.lang_id._datasets.Ted2020(src_root_dir.joinpath("ted2020"))
    ted2020.download(force=force)
    ted2020_data = ted2020.load(valid_langs, min_len=min_text_len)

    setimes = textacy.lang_id._datasets.SETimes(src_root_dir.joinpath("setimes"))
    setimes.download(force=force)
    setimes_data = setimes.load(valid_langs, min_len=min_text_len)

    ud = textacy.lang_id._datasets.UDDataset(src_root_dir.joinpath("ud"))
    ud.download(force=force)
    ud_data = ud.load(valid_langs, min_len=min_text_len)

    # aggregate and sample datasets
    agg_data = (
        udhr_data  # only has ~12k examples
        + get_random_sample(wili_data, len(wili_data), stratify=True, random_state=seed)
        + get_random_sample(tatoeba_data, 2_500_000, stratify=True, random_state=seed)
        + get_random_sample(ted2020_data, 2_500_000, stratify=True, random_state=seed)
        + get_random_sample(ud_data, 2_500_000, stratify=True, random_state=seed)
        # add additional examples for hard-to-distinguish language groups
        + get_random_sample(dslcc_data, 100_000, stratify=True, random_state=seed)
        + get_random_sample(setimes_data, 200_000, stratify=True, random_state=seed)
    )

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
    text_lens = tuple(len(text) for text, _ in data)
    print(
        f"min text len: {min(text_lens)}\n"
        f"mean text len: {statistics.mean(text_lens)}\n"
        f"stdev text len: {statistics.stdev(text_lens)}\n"
        f"max text len: {max(text_lens)}"
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


if __name__ == "__main__":
    main()
