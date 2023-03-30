from __future__ import annotations

import argparse
import logging
import pathlib

import floret
import sklearn.metrics

import textacy
import textacy.lang_id

logging.basicConfig(level=logging.INFO)


def main():
    args = add_and_parse_args()
    root_dirpath: pathlib.Path = args.root_dirpath.resolve()
    test_fpath = root_dirpath / "test.txt"
    lang_identifier = textacy.lang_id.LangIdentifier(
        version=args.version, data_dir=root_dirpath
    )

    logging.info("training language identifier model ...")
    model = floret.train_supervised(
        str(root_dirpath / "train.txt"),
        dim=args.dim,
        minn=args.minn,
        maxn=args.maxn,
        wordNgrams=args.wordNgrams,
        lr=args.lr,
        loss=args.loss,
        epoch=args.epoch,
        thread=args.thread,
    )
    if args.cutoff:
        logging.info("compressing language identifier model ...")
        model.quantize(
            str(root_dirpath / "train.txt"),
            cutoff=args.cutoff,
            retrain=True,
            qnorm=True,
            dsub=2,
            verbose=True,
        )

    lang_identifier._model = model
    # lang_identifier.load_model()  # HACK! to skip training and just do eval

    eval_report = _evaluate_model(test_fpath, lang_identifier)
    print(eval_report)

    if args.save:
        lang_identifier.save_model()


def add_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Thin wrapper around floret/fasttext's `train_supervised` function.",
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
        help="semantic version number to assign to trained model, e.g. '3.0'",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--minn", type=int, default=1)
    parser.add_argument("--maxn", type=int, default=5)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--loss", type=str, default="hs")
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--thread", type=int, default=None)
    parser.add_argument("--cutoff", type=int, required=False, default=350_000)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="if specified, force downloads of all datasets, "
        "even if they already exist on disk under ``root_dirpath``",
    )
    return parser.parse_args()


def _evaluate_model(
    test_fpath: pathlib.Path, lang_identifier: textacy.lang_id.LangIdentifier
) -> str:
    logging.info("evaluating model on test data at %s ...", test_fpath)
    with test_fpath.open("r") as f:
        lines = (line.strip() for line in f)
        label_texts = (line.split(" ", maxsplit=1) for line in lines)
        labels, texts = tuple(zip(*label_texts))

    # using fasttext's underlying "multiline predict" should be faster than our python
    # pred_labels = tuple(lang_identifier.identify_lang(text) for text in texts)
    pred_labels, _ = lang_identifier.model.predict(list(texts), k=1)

    report = sklearn.metrics.classification_report(
        [lang_identifier._to_lang(label) for label in labels],
        [lang_identifier._to_lang(pred_label[0]) for pred_label in pred_labels],
    )
    assert isinstance(report, str)  # type guard
    return report

    # yes, floret/fasttext has functionality for model evaluation
    # but it's not nearly so nice as sklearn's
    # label_prfs = model.test_label(str(root_dirpath / "test.txt"), k=1)
    # print(
    #     "\n".join(
    #         f"{x[0].removeprefix('__label__')}: {x[1]['f1score']:.2f}"
    #         for x in sorted(label_prfs.items(), key=lambda x: x[0])
    #     )
    # )


if __name__ == "__main__":
    main()
