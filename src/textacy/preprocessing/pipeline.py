from typing import Callable

from cytoolz import functoolz


def make_pipeline(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """
    Make a callable pipeline that takes a text string as input, passes it through
    one or multiple functions in the order with which they were specified, then outputs
    a single (preprocessed) text string.

    This function is intended as a lightweight convenience for users, allowing them
    to flexibly specify which (and in which order) preprocessing functions are to be
    applied to raw texts, then treating the whole thing as a single callable.

    .. code-block:: pycon

        >>> from textacy import preprocessing
        >>> preproc = preprocessing.make_pipeline(
        ...     preprocessing.remove.accents,
        ...     preprocessing.replace.phone_numbers,
        ...     preprocessing.normalize.whitespace,
        ... )
        >>> preproc("Mi número de teléfono es 555-123-4567.\n\n¿Cuál es el suyo?")
        'Mi numero de telefono es _PHONE_.\nCual es el suyo?'
        >>> preproc("¿Quién tiene un celular?  Tengo que llamar a mi mamá.")
        'Quien tiene un celular? Tengo que llamar a mi mama.'

    Args:
        *funcs

    Returns:
        Pipeline composed of ``*funcs`` that applies each in sequential order.
    """
    return functoolz.compose_left(*funcs)
