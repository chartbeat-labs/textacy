from __future__ import absolute_import, division, print_function, unicode_literals

import math

import networkx as nx

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


RC_PARAMS = {
    "axes.axisbelow": True,
    "axes.edgecolor": ".8",
    "axes.facecolor": "white",
    "axes.grid": False,
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


def draw_semantic_network(
    graph,
    node_weights=None,
    spread=3.0,
    draw_nodes=False,
    base_node_size=300,
    node_alpha=0.25,
    line_width=0.5,
    line_alpha=0.1,
    base_font_size=12,
    save=False,
):
    """
    Draw a semantic network with nodes representing either terms or sentences,
    edges representing coocurrence or similarity, and positions given by a force-
    directed layout.

    Args:
        graph (``networkx.Graph``):
        node_weights (dict): mapping of node: weight, used to size node labels
            (and, optionally, node circles) according to their weight
        spread (float): number that drives the spread of the network; higher
            values give more spread-out networks
        draw_nodes (bool): if True, circles are drawn under the node labels
        base_node_size (int): if `node_weights` not given and `draw_nodes` is True,
            this is the size of all nodes in the network; if `node_weights` _is_
            given, node sizes will be scaled against this value based on their
            weights compared to the max weight
        node_alpha (float): alpha of the circular nodes drawn behind labels if
            `draw_nodes` is True
        line_width (float): width of the lines (edges) drawn between nodes
        line_alpha (float): alpha of the lines (edges) drawn between nodes
        base_font_size (int): if `node_weights` not given, this is the font size
            used to draw all labels; otherwise, font sizes will be scaled against
            this value based on the corresponding node weights compared to the max
        save (str): give the full /path/to/fname on disk to save figure (optional)

    Returns:
        :obj:`matplotlib.axes.Axes.axis`: Axis on which network plot is drawn.

    Note:
        This function requires `matplotlib <https://matplotlib.org/>`_.
    """
    try:
        plt
    except NameError:
        raise ImportError(
            "`matplotlib` is not installed, so `textacy.viz` won't work; "
            "install it individually via `$ pip install matplotlib`, or "
            "along with textacy via `pip install textacy[viz]`."
        )
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(12, 12))

        pos = nx.layout.spring_layout(graph, k=spread / math.sqrt(len(graph.nodes())))
        _ = nx.draw_networkx_edges(
            graph, ax=ax, pos=pos, width=line_width, alpha=line_alpha, arrows=False
        )

        if node_weights is None:
            if draw_nodes is True:
                _ = nx.draw_networkx_nodes(
                    graph,
                    ax=ax,
                    pos=pos,
                    alpha=node_alpha,
                    linewidths=0.5,
                    node_size=base_node_size,
                )
            _ = nx.draw_networkx_labels(
                graph,
                pos,
                ax=ax,
                font_size=base_font_size,
                font_color="black",
                font_family="sans-serif",
            )
        else:
            max_node_weight = max(node_weights.values())
            if draw_nodes is True:
                node_sizes = [
                    base_node_size * pow(node_weights[node] / max_node_weight, 0.75)
                    for node in graph.nodes()
                ]
                _ = nx.draw_networkx_nodes(
                    graph,
                    ax=ax,
                    pos=pos,
                    node_size=node_sizes,
                    alpha=node_alpha,
                    linewidths=0.5,
                )
            for node, weight in node_weights.items():
                _ = nx.draw_networkx_labels(
                    graph,
                    pos,
                    labels={node: node},
                    ax=ax,
                    font_color="black",
                    font_family="sans-serif",
                    font_size=base_font_size * pow(weight / max_node_weight, 0.15),
                )

        ax.set_frame_on(False)
        ax.set_xticklabels(["" for _ in range(len(ax.get_xticklabels()))])
        ax.set_yticklabels(["" for _ in range(len(ax.get_yticklabels()))])

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=100)

    return ax
