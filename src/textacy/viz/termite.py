import numpy as np

from .. import errors
from ..utils import to_collection

try:
    import matplotlib.pyplot as plt
    import matplotlib.transforms

except ImportError:
    pass


RC_PARAMS = {
    "axes.axisbelow": True,
    "axes.edgecolor": ".8",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.labelcolor": ".15",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "sans-serif"],
    "grid.color": ".8",
    "grid.linestyle": "-",
    "image.cmap": "Greys",
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "lines.solid_capstyle": "round",
    "text.color": ".15",
    "xtick.color": ".15",
    "xtick.direction": "out",
    "xtick.major.size": 0.0,
    "xtick.minor.size": 0.0,
    "ytick.color": ".15",
    "ytick.direction": "out",
    "ytick.major.size": 0.0,
    "ytick.minor.size": 0.0,
}

COLOR_PAIRS = (
    (
        (0.65098041296005249, 0.80784314870834351, 0.89019608497619629),
        (0.12572087695201239, 0.47323337360924367, 0.707327968232772),
    ),
    (
        (0.68899655751153521, 0.8681737867056154, 0.54376011946622071),
        (0.21171857311445125, 0.63326415104024547, 0.1812226118410335),
    ),
    (
        (0.98320646005518297, 0.5980161709820524, 0.59423301088459368),
        (0.89059593116535862, 0.10449827132271793, 0.11108035462744099),
    ),
    (
        (0.99175701702342312, 0.74648213716698619, 0.43401768935077328),
        (0.99990772780250103, 0.50099192647372981, 0.0051211073118098693),
    ),
    (
        (0.78329874347238004, 0.68724338552531095, 0.8336793640080622),
        (0.42485198495434734, 0.2511495584950722, 0.60386007743723258),
    ),
    (
        (0.99760092286502611, 0.99489427150464516, 0.5965244373854468),
        (0.69411766529083252, 0.3490196168422699, 0.15686275064945221),
    ),
)


def draw_termite_plot(
    values_mat,
    col_labels,
    row_labels,
    *,
    highlight_cols=None,
    highlight_colors=None,
    save=False,
    rc_params=None,
):
    """
    Make a "termite" plot, typically used for assessing topic models with a tabular
    layout that promotes comparison of terms both within and across topics.

    Args:
        values_mat (:class:`np.ndarray` or matrix): matrix of values with shape
            (# row labels, # col labels) used to size the dots on the grid
        col_labels (seq[str]): labels used to identify x-axis ticks on the grid
        row_labels(seq[str]): labels used to identify y-axis ticks on the grid
        highlight_cols (int or seq[int], optional): indices for columns
            to visually highlight in the plot with contrasting colors
        highlight_colors (tuple of 2-tuples): each 2-tuple corresponds to a pair
            of (light/dark) matplotlib-friendly colors used to highlight a single
            column; if not specified (default), a good set of 6 pairs are used
        save (str, optional): give the full /path/to/fname on disk to save figure
        rc_params (dict, optional): allow passing parameters to rc_context in matplotlib.plyplot,
            details in https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.rc_context.html

    Returns:
        :obj:`matplotlib.axes.Axes.axis`: Axis on which termite plot is plotted.

    Raises:
        ValueError: if more columns are selected for highlighting than colors
            or if any of the inputs' dimensions don't match

    References:
        Chuang, Jason, Christopher D. Manning, and Jeffrey Heer. "Termite:
        Visualization techniques for assessing textual topic models."
        Proceedings of the International Working Conference on Advanced
        Visual Interfaces. ACM, 2012.

    See Also:
        :meth:`TopicModel.termite_plot() <textacy.tm.topic_model.TopicModel.termite_plot>`
    """
    try:
        plt
    except NameError:
        raise ImportError(
            "`matplotlib` is not installed, so `textacy.viz` won't work; "
            "install it individually via `$ pip install matplotlib`, or "
            "along with textacy via `pip install textacy[viz]`."
        )
    n_rows, n_cols = values_mat.shape
    max_val = np.max(values_mat)

    if n_rows != len(row_labels):
        raise ValueError(
            "values_mat and row_labels dimensions don't match: "
            f"{n_rows} vs. {len(row_labels)}"
        )
    if n_cols != len(col_labels):
        raise ValueError(
            "values_mat and col_labels dimensions don't match: "
            f"{n_cols} vs. {len(col_labels)}"
        )

    if highlight_colors is None:
        highlight_colors = COLOR_PAIRS
    if highlight_cols is not None:
        if isinstance(highlight_cols, int):
            highlight_cols = (highlight_cols,)
        elif len(highlight_cols) > len(highlight_colors):
            raise ValueError(
                f"no more than {len(highlight_colors)} columns "
                "may be highlighted at once"
            )

        highlight_colors = {hc: COLOR_PAIRS[i] for i, hc in enumerate(highlight_cols)}

    _rc_params = RC_PARAMS.copy()
    if rc_params:
        _rc_params.update(rc_params)

    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(pow(n_cols, 0.8), pow(n_rows, 0.66)))

        _ = ax.set_yticks(range(n_rows))
        yticklabels = ax.set_yticklabels(row_labels, fontsize=14, color="gray")
        if highlight_cols is not None:
            for i, ticklabel in enumerate(yticklabels):
                max_tick_val = max(values_mat[i, hc] for hc in range(n_cols))
                for hc in highlight_cols:
                    if max_tick_val > 0 and values_mat[i, hc] == max_tick_val:
                        ticklabel.set_color(highlight_colors[hc][1])

        ax.get_xaxis().set_ticks_position("top")
        _ = ax.set_xticks(range(n_cols))
        xticklabels = ax.set_xticklabels(
            col_labels, fontsize=14, color="gray", rotation=-60, ha="right"
        )

        # Create offset transform by 5 points in x direction
        dx = 10 / 72
        dy = 0
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        if highlight_cols is not None:
            gridlines = ax.get_xgridlines()
            for i, ticklabel in enumerate(xticklabels):
                if i in highlight_cols:
                    ticklabel.set_color(highlight_colors[i][1])
                    gridlines[i].set_color(highlight_colors[i][0])
                    gridlines[i].set_alpha(0.5)

        for col_ind in range(n_cols):
            if highlight_cols is not None and col_ind in highlight_cols:
                ax.scatter(
                    [col_ind for _ in range(n_rows)],
                    [i for i in range(n_rows)],
                    s=600 * (values_mat[:, col_ind] / max_val),
                    alpha=0.5,
                    linewidth=1,
                    color=highlight_colors[col_ind][0],
                    edgecolor=highlight_colors[col_ind][1],
                )
            else:
                ax.scatter(
                    [col_ind for _ in range(n_rows)],
                    [i for i in range(n_rows)],
                    s=600 * (values_mat[:, col_ind] / max_val),
                    alpha=0.5,
                    linewidth=1,
                    color="lightgray",
                    edgecolor="gray",
                )

            _ = ax.set_xlim(left=-1, right=n_cols)
            _ = ax.set_ylim(bottom=-1, top=n_rows)

            ax.invert_yaxis()  # otherwise, values/labels go from bottom to top

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=100)

    return ax


def termite_df_plot(
    components,
    *,
    highlight_topics=None,
    n_terms=25,
    rank_terms_by="max",
    sort_terms_by="seriation",
    save=False,
    rc_params=None,
):
    """
    Make a "termite" plot for assessing topic models using a tabular layout
    to promote comparison of terms both within and across topics.

    Args:
        components (:class:`pandas.DataFrame` or sparse matrix): corpus
            represented as a term-topic matrix with shape (n_terms, n_topics);
            should have terms as index and topics as column names
        topics (int or Sequence[int]): topic(s) to include in termite plot;
            if -1, all topics are included
        highlight_topics (str or Sequence[str]): names for up to 6 topics
            to visually highlight in the plot with contrasting colors
        n_terms (int): number of top terms to include in termite plot
        rank_terms_by ({'max', 'mean', 'var'}): argument to dataframe `agg`
            function, used to rank terms;
            the top-ranked ``n_terms`` are included in the plot
        sort_terms_by ({'seriation', 'weight', 'index', 'alphabetical'}):
            method used to vertically sort the selected top ``n_terms`` terms;
            the default ("seriation") groups similar terms together, which
            facilitates cross-topic assessment
        save (str): give the full /path/to/fname on disk to save figure
            rc_params (dict, optional): allow passing parameters to rc_context
            in matplotlib.plyplot, details in
            https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.rc_context.html

    Returns:
        ``matplotlib.axes.Axes.axis``: Axis on which termite plot is plotted.

    Raises:
        ValueError: if more than 6 topics are selected for highlighting, or
            an invalid value is passed for the sort_topics_by, rank_terms_by,
            and/or sort_terms_by params

    References:
        - Chuang, Jason, Christopher D. Manning, and Jeffrey Heer. "Termite:
            Visualization techniques for assessing textual topic models."
            Proceedings of the International Working Conference on Advanced
            Visual Interfaces. ACM, 2012.
        - Fajwel Fogel, Alexandre d’Aspremont, and Milan Vojnovic. 2016.
            Spectral ranking using seriation. J. Mach. Learn. Res. 17, 1
            (January 2016), 3013–3057.

    See Also:
        :func:`viz.termite_plot <textacy.viz.termite.termite_plot>`
    TODO: `rank_terms_by` other metrics, e.g. topic salience or relevance
    """
    highlight_topics = to_collection(highlight_topics, str, list)

    if len(highlight_topics) > 6:
        raise ValueError("no more than 6 topics may be highlighted at once")

    # get column index of any topics to highlight in termite plot
    if highlight_topics is not None:
        highlight_cols = tuple(components.columns.get_loc(c) for c in highlight_topics)
    else:
        highlight_cols = None

    component_filter = components.loc[
        components.agg(rank_terms_by, axis=1)
        .sort_values(ascending=False)
        .iloc[:n_terms]
        .index
    ]

    # get top term indices in sorted order
    if sort_terms_by == "weight":
        pass  # already done
    elif sort_terms_by == "alphabetical":
        component_filter = component_filter.sort_index()
    elif sort_terms_by == "seriation":
        # calculate similarity matrix
        similarity = (
            component_filter @ component_filter.T.pipe(lambda df: df - df.min().min())
        ).values

        # compute Laplacian matrice and its 2nd eigenvector
        L = np.diag(similarity.sum(axis=1)) - similarity
        D, V = np.linalg.eigh(L)
        D = D[np.argsort(D)]
        V = V[:, np.argsort(D)]
        fiedler = V[:, 1]

        # get permutation corresponding to sorting the 2nd eigenvector
        component_filter = component_filter.reindex(
            index=[component_filter.index[i] for i in np.argsort(fiedler)],
        )
    else:
        raise ValueError(
            errors.value_invalid_msg(
                "sort_terms_by", sort_terms_by, {"weight", "alphabetical", "seriation"},
            )
        )

    # get topic and term labels
    topic_labels = component_filter.columns
    term_labels = component_filter.index

    # get topic-term weights to size dots
    term_topic_weights = component_filter.values

    return draw_termite_plot(
        term_topic_weights,
        topic_labels,
        term_labels,
        highlight_cols=highlight_cols,
        save=save,
        rc_params=rc_params,
    )
