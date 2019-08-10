"""
Collection of regular expressions and other (small, generally useful) constants.
"""
import pathlib
import re


DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"

NUMERIC_ENT_TYPES = {"ORDINAL", "CARDINAL", "MONEY", "QUANTITY", "PERCENT", "TIME", "DATE"}
SUBJ_DEPS = {"agent", "csubj", "csubjpass", "expl", "nsubj", "nsubjpass"}
OBJ_DEPS = {"attr", "dobj", "dative", "oprd"}
AUX_DEPS = {"aux", "auxpass", "neg"}

REPORTING_VERBS = {
    "according", "accuse", "acknowledge", "add", "admit", "agree",
    "allege", "announce", "argue", "ask", "assert", "believe", "blame",
    "charge", "cite", "claim", "complain", "concede", "conclude",
    "confirm", "contend", "criticize", "declare", "decline", "deny",
    "describe", "disagree", "disclose", "estimate", "explain", "fear",
    "hope", "insist", "maintain", "mention", "note", "observe", "order",
    "predict", "promise", "recall", "recommend", "reply", "report", "say",
    "state", "stress", "suggest", "tell", "testify", "think", "urge", "warn",
    "worry", "write"
}

MATCHER_VALID_OPS = {"!", "+", "?", "*"}

POS_REGEX_PATTERNS = {
    "en": {
        "NP": r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+",
        "PP": r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+",
        "VP": r"<AUX>* <ADV>* <VERB>",
    }
}

RE_MATCHER_TOKPAT_DELIM = re.compile(r"\s+")
RE_MATCHER_SPECIAL_VAL = re.compile(
    r"^(int|bool)\([^: ]+\)$",
    flags=re.UNICODE)

RE_ACRONYM = re.compile(
    r"(?:^|(?<=\W))"
    r"(?:"
    r"(?:(?:(?:[A-Z]\.?)+[a-z0-9&/-]?)+(?:[A-Z][s.]?|\ds?))"
    r"|"
    r"(?:\d(?:\-?[A-Z])+)"
    r")"
    r"(?:$|(?=\W))",
    flags=re.UNICODE)

RE_LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
RE_NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

# regexes for cleaning up crufty terms
RE_DANGLING_PARENS_TERM = re.compile(
    r"(?:\s|^)(\()\s{1,2}(.*?)\s{1,2}(\))(?:\s|$)", flags=re.UNICODE)
RE_LEAD_TAIL_CRUFT_TERM = re.compile(
    r"^([^\w(-] ?)+|([^\w).!?] ?)+$", flags=re.UNICODE)
RE_LEAD_HYPHEN_TERM = re.compile(
    r"^-([^\W\d_])", flags=re.UNICODE)
RE_NEG_DIGIT_TERM = re.compile(
    r"(-) (\d)", flags=re.UNICODE)
RE_WEIRD_HYPHEN_SPACE_TERM = re.compile(
    r"(?<=[^\W\d]) (-[^\W\d])", flags=re.UNICODE)
RE_WEIRD_APOSTR_SPACE_TERM = re.compile(
    r"([^\W\d]+) ('[a-z]{1,2}\b)", flags=re.UNICODE)
