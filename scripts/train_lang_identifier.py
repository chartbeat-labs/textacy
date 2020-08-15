"""
### env

textacy==0.9.0
...

### fetch source data

- **Tatoeba:** A crowd-sourced collection of sentences and their translations into many languages. Style is relatively informal; subject matter is a variety of everyday things and goings-on. Source: https://tatoeba.org/eng/downloads.
- **Leipzig Corpora:** A collection of corpora for many languages pulling from comparable sources -- specifically, 10k Wikipedia articles from official database dumps and 10k news articles from either RSS feeds or web scrapes, when available. Style is relatively formal; subject matter is a variety of notable things and goings-on. Source: http://wortschatz.uni-leipzig.de/en/download
- **UDHR:** The UN's Universal Declaration of Human Rights document, translated into hundreds of languages and split into paragraphs. Style is formal; subject matter is fundamental human rights to be universally protected. Source: https://unicode.org/udhr/index.html
- **Twitter:** A collection of tweets in each of ~70 languages, posted in July 2014, with languages assigned through a combination of models and human annotators. Style is informal; subject matter is whatever Twitter was going on about back then, who could say. Source: https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html
- **DSLCC**: Two collections of short excerpts of journalistic texts in a handful of language groups that are highly similar to each other. Style is relatively formal; subject matter is current events. Source: http://ttg.uni-saarland.de/resources/DSLCC/
"""
import argparse
import collections
import logging
import operator
import os
import random
import re
import statistics
import sys
import tarfile
from pprint import pprint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import sklearn.model_selection
import tqdm
import twitter  # python-twitter on pypi
import yaml
from cytoolz import itertoolz

import textacy
import textacy.datasets
import textacy.io
import textacy.preprocessing

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_dirpath",
        required=True,
        help="path to root directory under which datasets and models are saved",
    )
    parser.add_argument(
        "--twitter_creds",
        required=True,
        help="path to file containing twitter API credentials",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=25,
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
        "--version",
        type=str,
        required=True,
        help="semantic version number to assign to trained model, e.g. '1.0'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="if specified, force downloads of all datasets, "
        "even if they already exist on disk under ``root_dirpath``",
    )
    args = parser.parse_args()

    root_dirpath = textacy.utils.to_path(args.root_dirpath).resolve()
    iso_639_dirpath = root_dirpath.joinpath("iso-639")
    twitter_dirpath = root_dirpath.joinpath("twitter")
    tatoeba_dirpath = root_dirpath.joinpath("tatoeba")
    wili_dirpath = root_dirpath.joinpath("wili")
    udhr_dirpath = root_dirpath.joinpath("udhr")
    dslcc_dirpath = root_dirpath.joinpath("dslcc")
    models_dirpath = root_dirpath.joinpath("models")

    # download, load, and summarize datasets
    download_iso_639_data(iso_639_dirpath, force=args.force)
    iso_639_data = load_iso_639_data(iso_639_dirpath, exclude=("sh",))
    print(sorted(iso_639_data.items())[:5], "...")

    download_twitter_data(twitter_dirpath, args.twitter_creds, force=args.force)
    twitter_data = load_twitter_data(
        twitter_dirpath, set(iso_639_data.values()), min_len=args.min_len
    )
    summarize_dataset(twitter_data)

    download_tatoeba_data(tatoeba_dirpath, force=args.force)
    tatoeba_data = load_tatoeba_data(tatoeba_dirpath, iso_639_data, min_len=args.min_len)
    summarize_dataset(tatoeba_data)

    download_wili_data(wili_dirpath, force=args.force)
    wili_data = load_wili_data(wili_dirpath, iso_639_data, min_len=args.min_len)
    summarize_dataset(wili_data)

    download_udhr_data(udhr_dirpath, force=args.force)
    udhr_data = load_udhr_data(
        udhr_dirpath, set(iso_639_data.values()), min_len=args.min_len
    )
    summarize_dataset(udhr_data)

    download_dslcc_data(dslcc_dirpath, force=args.force)
    dslcc_data = load_dslcc_data(
        dslcc_dirpath, set(iso_639_data.values()), min_len=args.min_len
    )
    summarize_dataset(dslcc_data)

    # HACK HACK HACK
    leipzig_dirpath = textacy.utils.to_path(
        "/Users/burtondewilde/Desktop/datasets/language_identification/leipzig-corpora"
    ).resolve()
    if leipzig_dirpath.is_dir():
        leipzig_data = load_leipzig_data(
            leipzig_dirpath, iso_639_data, min_len=args.min_len
        )
        summarize_dataset(leipzig_data)
    else:
        logging.warning("leipzig data hack unavailable, sorry")
        leipzig_data = []

    # aggregate and sample datasets
    datasets = (
        udhr_data
        + wili_data
        + get_random_sample(tatoeba_data, 420000, stratify=True, random_state=42)
        + get_random_sample(leipzig_data, 480000, stratify=True, random_state=42)
        + get_random_sample(
            twitter_data, len(twitter_data), stratify=True, random_state=42
        )
        + get_random_sample(dslcc_data, 20000, stratify=True, random_state=42)
        + get_random_sample(
            [item for item in tatoeba_data if item[1] == "en"],
            10000,
            stratify=False,
            random_state=42,
        )
    )
    datasets = get_random_sample(datasets, 3000000, stratify=True)

    valid_langs = {
        lang
        for lang, count in collections.Counter(
            lang for _, lang in datasets
        ).most_common()
        if count >= args.min_obs
    }
    datasets = [text_lang for text_lang in datasets if text_lang[1] in valid_langs]

    chars_set = {char for text, _ in datasets for char in text}
    langs_set = {lang for _, lang in datasets}
    num_langs = len(langs_set)
    print("# unique chars:", len(chars_set))
    print("# unique langs:", num_langs)
    summarize_dataset(datasets)

    train_items, test_items = sklearn.model_selection.train_test_split(
        datasets, test_size=0.2, random_state=42
    )
    print(train_items[0])
    print("# train items:", len(train_items))
    print("# test items:", len(test_items))

    # fit and validate a model
    model_id = "lang-identifier-v{}-sklearn-v{}".format(
        args.version, sklearn.__version__[:4]
    )
    print("training model {} ...".format(model_id))
    lid = textacy.lang_utils.LangIdentifier(data_dir=models_dirpath)
    lid.init_pipeline()
    print(lid.pipeline.steps)
    lid.pipeline.fit(
        [text for text, _ in train_items], y=[lang for _, lang in train_items],
    )
    true, preds = test_model(
        lid.pipeline, test_items, filepath=models_dirpath.joinpath(model_id + ".txt")
    )
    try:
        ax = plot_confusion_matrix(
            true, preds, normalize=True, title=None, cmap=plt.cm.Blues, annotate=False
        )
        ax.get_figure().savefig(
            models_dirpath.joinpath(model_id + "-confusion-matrix.png")
        )
    except Exception:
        pass  # well, at least we tried
    joblib.dump(lid.pipeline, models_dirpath.joinpath(model_id + ".pkl.gz"), compress=3)


def download_iso_639_data(dirpath, force=False):
    """
    Download official ISO 639 code table as a TSV,
    mapping all language code variations (639-1, 639-2, 639-3)
    to each other.

    Args:
        dirpath (str or :class:`pathlib.Path`)
        force (bool)

    References:
        https://iso639-3.sil.org/code_tables/639/data
    """
    url = "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab"
    textacy.io.download_file(url, filename="iso-639-3.tsv", dirpath=dirpath, force=force)


def load_iso_639_data(dirpath, exclude=None):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        exclude (Set[str])

    Returns:
        Dict[str, str]
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    rows = textacy.io.read_csv(
        dirpath.joinpath("iso-639-3.tsv").resolve(),
        delimiter="\t",
        fieldnames=[
            "Id",
            "Part2B",
            "Part2T",
            "Part1",
            "Scope",
            "Language_Type",
            "Ref_Name",
            "Comment",
        ],
        quoting=1,
    )
    lang_map = {
        row["Id"]: row["Part1"]
        for row in rows
        if row.get("Part1") and (exclude is None or row["Part1"] not in exclude)
    }
    return lang_map


def download_twitter_data(dirpath, creds_fpath, force=False):
    """
    Download two collections of (lang, tweet_id) pairs from Twitter --
    a "uniformly sampled" collection of ~120k tweets over all languages and
    a "recall oriented" collection of ~1.5k tweets per language --
    then fetch available tweets' data from the Twitter API.

    Args:
        dirpath (str or :class:`pathlib.Path`)
        creds_fpath (str or :class:`pathlib.Path`)
        force (bool)

    References:
        https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html

    TODO: Ideally, use a tweet search endpoint and filter by language,
    then just iterate over all ISO-639-1 language codes.
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    url_fnames = [
        (
            "https://raw.githubusercontent.com/mitjat/langid_eval/master/uniformly_sampled.tsv",
            "uniformly_sampled.tsv",
        ),
        (
            "https://raw.githubusercontent.com/mitjat/langid_eval/master/recall_oriented.tsv",
            "recall_oriented.tsv",
        ),
    ]
    # download tweet ids first
    for url, fname in url_fnames:
        textacy.io.download_file(url, filename=fname, dirpath=dirpath, force=force)
    # download full tweets data next
    tweets_fpath = dirpath.joinpath("tweets.jsonl")
    if tweets_fpath.is_file() and force is False:
        logging.info("tweets data already downloaded to %s", tweets_fpath)
        return

    # load twitter ids data from disk
    tweet_lang_ids = []
    for fname in ["uniformly_sampled.tsv", "recall_oriented.tsv"]:
        tweet_lang_ids.extend(
            textacy.io.read_csv(
                dirpath.joinpath(fname),
                delimiter="\t",
                fieldnames=["lang", "status_id"],
                quoting=1,
            )
        )
    logging.info("loaded %s tweet ids from disk", len(tweet_lang_ids))
    # parse status ids
    status_ids = set()
    for row in tweet_lang_ids:
        try:
            status_ids.add(int(row["status_id"]))
        # there are a small handful of bad status ids, shrug
        except ValueError:
            pass
    logging.info("... of which %s had valid, unique ids", len(status_ids))
    status_ids = list(status_ids)
    # instantiate twitter api client
    with textacy.utils.to_path(creds_fpath).resolve().open(mode="rt") as f:
        creds = yaml.safe_load(f.read())
    api = twitter.Api(sleep_on_rate_limit=True, **creds)
    # get tweets data in chunks
    chunk_size = 100
    pbar = tqdm.tqdm(total=len(status_ids), unit="tweets")
    tweets = []
    try:
        for chunk_ids in itertoolz.partition_all(chunk_size, status_ids):
            chunk_tweets = api.GetStatuses(
                chunk_ids, trim_user=True, include_entities=True, map=False
            )
            tweets.extend(chunk_tweets)
            pbar.update(len(chunk_ids))
    except Exception:
        logging.exception("encountered an error while downloading tweets")
    finally:
        pbar.close()
        tweets = [tweet.AsDict() for tweet in tweets]
        logging.info("downloaded data for %s tweets", len(tweets))
        textacy.io.write_json(tweets, tweets_fpath, mode="wt", lines=True)


def load_twitter_data(dirpath, langs, min_len=25):
    """
    Args:
        dirpath (str)
        langs (Set[str])
        min_len (int): minimum text length in *chars*

    Returns:
        List[Tuple[str, str]]
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    raw_tweets = textacy.io.read_json(
        dirpath.joinpath("tweets.jsonl"), mode="rt", lines=True
    )
    tweets = []
    for tweet in raw_tweets:
        # totally remove any URLS from tweet text
        for url in tweet.get("urls", []):
            for item in url.values():
                tweet["text"] = tweet["text"].replace(item, "")
        tweets.append(tweet)
    ds = [
        (tweet["text"], tweet["lang"])
        for tweet in tweets
        if tweet["lang"] in langs
        and itertoolz.count(char for char in tweet["text"] if char.isalnum()) >= min_len
    ]
    return ds


def download_tatoeba_data(dirpath, force=False):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        force (bool)

    References:
        https://tatoeba.org/eng/downloads
    """
    url = "http://downloads.tatoeba.org/exports/sentences.tar.bz2"
    fpath = textacy.io.download_file(url, dirpath=dirpath, force=force)
    if fpath:
        textacy.io.unpack_archive(fpath, extract_dir=dirpath)


def load_tatoeba_data(dirpath, iso_lang_map, min_len=25):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        iso_lang_map (Dict[str, str])
        min_len (int): minimum text length in *chars*

    Returns:
        List[Tuple[str, str]]
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    rows = textacy.io.read_csv(
        dirpath.joinpath("sentences.csv"),
        fieldnames=["sent_id", "iso-639-3", "text"],
        delimiter="\t",
        quoting=1,
    )
    langs = set(iso_lang_map.keys())
    ds = [
        (row["text"], iso_lang_map[row["iso-639-3"]])
        for row in rows
        if row["iso-639-3"] in langs
        and itertoolz.count(char for char in row["text"] if char.isalnum()) >= min_len
    ]
    return ds


def download_wili_data(dirpath, force=False):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        force (bool)

    References:
        https://tatoeba.org/eng/downloads
    """
    url = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"
    fpath = textacy.io.download_file(url, dirpath=dirpath, force=force)
    if fpath:
        textacy.io.unpack_archive(fpath, extract_dir=dirpath)


def load_wili_data(dirpath, iso_lang_map, min_len=25):
    """
    Args:
        dirpath (str)
        iso_lang_map (Dict[str, str])
        min_len (int): minimum text length in *chars*

    Returns:
        List[Tuple[str, str]]

    References:
        https://zenodo.org/record/841984
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    ds = []
    for subset in ("train", "test"):
        text_lines = textacy.io.read_text(
            dirpath.joinpath("x_{}.txt".format(subset)), lines=True
        )
        lang_lines = textacy.io.read_text(
            dirpath.joinpath("y_{}.txt".format(subset)), lines=True
        )
        texts = (line.strip() for line in text_lines)
        langs = (line.strip() for line in lang_lines)
        langs_set = set(iso_lang_map.keys())
        ds.extend(
            (text, iso_lang_map[lang])
            for text, lang in zip(texts, langs)
            if lang in langs_set
            and itertoolz.count(char for char in text if char.isalnum()) >= min_len
        )
    return ds


def download_udhr_data(dirpath, force=False):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        force (bool)
    """
    ds = textacy.datasets.UDHR(data_dir=dirpath)
    ds.download(force=force)


def load_udhr_data(dirpath, langs, min_len=25):
    """
    Args:
        dirpath (str or :class:`pathlib.Path`)
        langs (Set[str])
        min_len (int)

    Returns:
        List[Tuple[str, str]]
    """
    ds = textacy.datasets.UDHR(data_dir=dirpath)
    data = [
        (snippet, meta["lang"])
        for text, meta in ds.records()
        for snippet in text.split("\n")
        if meta["lang"] in langs
        and itertoolz.count(char for char in snippet if char.isalnum()) >= min_len
    ]
    return data


def download_dslcc_data(dirpath, force=False):
    """
    Download two multilingual collections of short excerpts of journalistic texts,
    focused on language groups that are very similar and thus more difficult
    to correctly identify.

    Args:
        dirpath (str or :class:`pathlib.Path`)
        force (bool)

    References:
        http://ttg.uni-saarland.de/resources/DSLCC/
    """
    dirpath = textacy.utils.to_path(dirpath).resolve()
    for version in [3, 4]:
        name = "dslcc{}".format(version)
        url = "http://scholar.harvard.edu/files/malmasi/files/{}.zip".format(name)
        fpath = textacy.io.download_file(url, dirpath=dirpath, force=force)
        if fpath:
            textacy.io.unpack_archive(fpath, extract_dir=dirpath.joinpath(name))


def load_dslcc_data(dirpath, langs, min_len=25):
    """
    Args:
        dirpath (str)
        langs (Set[str])
        min_len (int): minimum text length in *chars*

    Returns:
        List[Tuple[str, str]]
    """
    data = []
    fstubs = [
        "dslcc3/train/task1-train.txt",
        "dslcc3/train/task1-dev.txt",
        "dslcc4/DSL-TRAIN.txt",
        "dslcc4/DSL-DEV.txt",
    ]
    for fstub in fstubs:
        filepath = dirpath.joinpath(fstub)
        lines = textacy.io.read_text(filepath, mode="rt", encoding="utf-8", lines=True)
        for line in lines:
            if not line.strip():
                continue
            try:
                text, lang = line.split("\t")
                if len(text) >= min_len and lang[:2] in langs:
                    data.append((text, lang[:2]))
            except Exception:
                logging.warning("bad line in data")
                pass
    return sorted(set(data), key=operator.itemgetter(1))


def load_leipzig_data(dirpath, iso_lang_map, min_len=25):
    """
    Args:
        dirpath (str)
        iso_lang_map (Dict[str, str])
        min_len (int): minimum text length in *chars*

    Returns:
        List[Tuple[str, str]]

    References:
        http://wortschatz.uni-leipzig.de/en/download

    TODO: figure out a good way of downloading leipzig data programatically?
    TODO: only use news data, maybe?
    """
    ds = []
    for filepath in sorted(textacy.io.get_filepaths(dirpath, match_regex=r"\.tar\.gz$")):
        fname = os.path.basename(filepath)
        lang = fname.split("_")[0]
        try:
            lang = iso_lang_map[lang]
        except KeyError:
            continue
        with tarfile.open(filepath, mode="r") as tf:
            for member in tf:
                if re.search(r".*?-sentences\.txt$", member.name):
                    with tf.extractfile(member) as f:
                        for line in f:
                            idx, text = line.decode("utf-8").split(sep="\t", maxsplit=1)
                            text = textacy.preprocessing.normalize_whitespace(text)
                            if len(text) >= min_len:
                                ds.append((text.strip(), lang))
    return ds


def summarize_dataset(ds):
    """
    Args:
        ds (List[Tuple[str, str]])
    """
    n_obs = len(ds)
    print("# observations:", n_obs)
    pprint(ds[:3])
    min_text_len = min(len(text) for text, _ in ds)
    print("...")
    print("min text len:", min_text_len)
    try:
        mean_text_len = statistics.mean(len(text) for text, _ in ds)
        stdev_text_len = statistics.stdev(len(text) for text, _ in ds)
        print("mean text len: {:.1f} +/- {:.1f}".format(mean_text_len, stdev_text_len))
    except NameError:
        pass
    max_text_len = max(len(text) for text, _ in ds)
    print("max text len:", max_text_len)
    lang_counts = collections.Counter(lang for _, lang in ds)
    top_counts = ", ".join(
        "'{}': {}".format(lang, count) for lang, count in lang_counts.most_common(15)
    )
    bot_counts = ", ".join(
        "'{}': {}".format(lang, count)
        for lang, count in sorted(
            lang_counts.items(), key=operator.itemgetter(1), reverse=True
        )[-15:]
    )
    print("# languages:", len(lang_counts))
    print("{} ... {}".format(top_counts, bot_counts))


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


def test_model(model, ds_test, filepath=None):
    exceptions = collections.Counter()
    true = []
    preds = []
    with tqdm.tqdm(total=len(ds_test)) as pbar:
        for text, lang in ds_test:
            pbar.update(1)
            true.append(lang)
            try:
                lang = model.predict([text])[0]
                preds.append(lang)
            except Exception:
                exceptions.update([lang])
                preds.append("un")
    print("# exceptions :", len(exceptions))
    if len(exceptions):
        print(exceptions.most_common())
    classification_report = sklearn.metrics.classification_report(true, preds)
    print(classification_report)
    if filepath:
        filepath = textacy.utils.to_path(filepath).resolve()
        with filepath.open(mode="wt", encoding="utf-8") as f:
            f.write(classification_report)
    return true, preds


def plot_confusion_matrix(
    y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues, annotate=False
):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    classes = sklearn.utils.multiclass.unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, shrink=0.5)
    if title:
        ax.set_title(title)

    # formatting
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="true",
        xlabel="pred",
    )
    ax.tick_params(axis="x", which="major", labelrotation=90)
    ax.tick_params(axis="both", which="major", labelsize="x-small")

    # annotation
    if annotate is True:
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
    return ax


if __name__ == "__main__":
    sys.exit(main())
