# mypy: ignore-errors
# TODO: figure out typing on these DocExtFuncs that satisfies mypy
from .. import types
from ..spacier.extensions import doc_extensions_registry
from . import basics, counts, diversity, readability


@doc_extensions_registry.register("text_stats.basics")
def _get_doc_extensions_text_stats_basics() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "n_sents": {"getter": basics.n_sents},
        "n_words": {"getter": basics.n_words},
        "n_unique_words": {"getter": basics.n_unique_words},
        "n_chars_per_word": {"getter": basics.n_chars_per_word},
        "n_chars": {"getter": basics.n_chars},
        "n_long_words": {"getter": basics.n_long_words},
        "n_syllables_per_word": {"getter": basics.n_syllables_per_word},
        "n_syllables": {"getter": basics.n_syllables},
        "n_monosyllable_words": {"getter": basics.n_monosyllable_words},
        "n_polysyllable_words": {"getter": basics.n_polysyllable_words},
        "entropy": {"getter": basics.entropy},
    }


@doc_extensions_registry.register("text_stats.counts")
def _get_doc_extensions_text_stats_counts() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        "morph_counts": {"getter": counts.morph},
        "tag_counts": {"getter": counts.tag},
        "pos_counts": {"getter": counts.pos},
        "dep_counts": {"getter": counts.dep},
    }


@doc_extensions_registry.register("text_stats.diversity")
def _get_doc_extensions_text_stats_diversity() -> (
    dict[str, dict[str, types.DocExtFunc]]
):
    return {
        "ttr": {"method": diversity.ttr},
        "log_ttr": {"method": diversity.log_ttr},
        "segmented_ttr": {"method": diversity.segmented_ttr},
        "mtld": {"method": diversity.mtld},
        "hdd": {"method": diversity.hdd},
    }


@doc_extensions_registry.register("text_stats.readability")
def _get_doc_extensions_text_stats_readability() -> (
    dict[str, dict[str, types.DocExtFunc]]
):
    return {
        "automated_readability_index": {
            "method": readability.automated_readability_index
        },
        "automatic_arabic_readability_index": {
            "method": readability.automatic_arabic_readability_index
        },
        "coleman_liau_index": {"method": readability.coleman_liau_index},
        "flesch_kincaid_grade_level": {
            "method": readability.flesch_kincaid_grade_level
        },
        "flesch_reading_ease": {"method": readability.flesch_reading_ease},
        "gulpease_index": {"method": readability.gulpease_index},
        "gunning_fog_index": {"method": readability.gunning_fog_index},
        "lix": {"method": readability.lix},
        "mu_legibility_index": {"method": readability.mu_legibility_index},
        "perspicuity_index": {"method": readability.perspicuity_index},
        "smog_index": {"method": readability.smog_index},
        "wiener_sachtextformel": {"method": readability.wiener_sachtextformel},
    }


@doc_extensions_registry.register("text_stats")
def _get_doc_extensions_text_stats() -> dict[str, dict[str, types.DocExtFunc]]:
    return {
        **_get_doc_extensions_text_stats_basics(),
        **_get_doc_extensions_text_stats_counts(),
        **_get_doc_extensions_text_stats_diversity(),
        **_get_doc_extensions_text_stats_readability(),
    }
