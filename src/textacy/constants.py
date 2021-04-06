"""
Collection of regular expressions and other (small, generally useful) constants.
"""
import pathlib
import re
from typing import Dict, Pattern, Set


DEFAULT_DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "data"

NUMERIC_ENT_TYPES: Set[str] = {
    "ORDINAL",
    "CARDINAL",
    "MONEY",
    "QUANTITY",
    "PERCENT",
    "TIME",
    "DATE",
}
SUBJ_DEPS: Set[str] = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS: Set[str] = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS: Set[str] = {"aux", "auxpass", "neg"}

REPORTING_VERBS: Dict[str, Set[str]] = {
    "en": {
        "according",
        "accuse",
        "acknowledge",
        "add",
        "admit",
        "agree",
        "allege",
        "announce",
        "argue",
        "ask",
        "assert",
        "believe",
        "blame",
        "charge",
        "cite",
        "claim",
        "complain",
        "concede",
        "conclude",
        "confirm",
        "contend",
        "criticize",
        "declare",
        "decline",
        "deny",
        "describe",
        "disagree",
        "disclose",
        "estimate",
        "explain",
        "fear",
        "hope",
        "insist",
        "maintain",
        "mention",
        "note",
        "observe",
        "order",
        "predict",
        "promise",
        "recall",
        "recommend",
        "reply",
        "report",
        "say",
        "state",
        "stress",
        "suggest",
        "tell",
        "testify",
        "think",
        "urge",
        "warn",
        "worry",
        "write",
    },
    "es": {
        "afirmar",
        "anunciar",
        "añadir",
        "asegurar",
        "comentar",
        "confesar",
        "contestar",
        "decir",
        "escribir",
        "exclamar",
        "explicar",
        "gritar",
        "preguntar",
        "prometer",
        "quejar",
        "recordar",
    },
    "fr": {
        "affirmer",
        "ajouter",
        "annoncer",
        "conclure",
        "crier",
        "déclarer",
        "demander",
        "dire",
        "écrire",
        "exclamer",
        "expliquer",
        "insister",
        "lancer",
        "mentionner",
        "prétendre",
        "proclamer",
        "prononcer",
        "proposer",
        "remarquer",
        "répliquer",
        "répondre",
        "rétorquer",
        "soutenir",
        "suggérer",
    },
}

MATCHER_VALID_OPS: Set[str] = {"!", "+", "?", "*"}

POS_REGEX_PATTERNS: Dict[str, Dict[str, str]] = {
    "en": {
        "NP": r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+",
        "PP": r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+",
        "VP": r"<AUX>* <ADV>* <VERB>",
    }
}

RE_MATCHER_TOKPAT_DELIM: Pattern = re.compile(r"\s+")
RE_MATCHER_SPECIAL_VAL: Pattern = re.compile(r"^(int|bool)\([^: ]+\)$", flags=re.UNICODE)

RE_ACRONYM: Pattern = re.compile(
    r"(?:^|(?<=\W))"
    r"(?:"
    r"(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|\ds?))"
    r"|"
    r"(?:\d(?:\-?[A-Z])+)"
    r")"
    r"(?:$|(?=\W))",
    flags=re.UNICODE,
)

RE_LINEBREAK: Pattern = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE: Pattern = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

# regexes for cleaning up crufty terms
RE_DANGLING_PARENS_TERM: Pattern = re.compile(
    r"(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)", flags=re.UNICODE
)
RE_LEAD_TAIL_CRUFT_TERM: Pattern = re.compile(r"^[^\w(-]+|[^\w).!?]+$", flags=re.UNICODE)
RE_LEAD_HYPHEN_TERM: Pattern = re.compile(r"^-([^\W\d_])", flags=re.UNICODE)
RE_NEG_DIGIT_TERM: Pattern = re.compile(r"(-) (\d)", flags=re.UNICODE)
RE_WEIRD_HYPHEN_SPACE_TERM: Pattern = re.compile(
    r"(?<=[^\W\d]) (-[^\W\d])", flags=re.UNICODE
)
RE_WEIRD_APOSTR_SPACE_TERM: Pattern = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE
)

RE_ALNUM: Pattern = re.compile(r"[^\W_]+")
