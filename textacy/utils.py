import warnings


def deprecated(message, action='always'):
    """
    Show a deprecation warning, optionally filtered.

    Args:
        message (str): Message to display with ``DeprecationWarning``.
        action (str): Filter controlling whether warning is ignored, displayed, or
            turned into an error. https://docs.python.org/3/library/warnings.html#the-warnings-filter
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action, DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=2)
