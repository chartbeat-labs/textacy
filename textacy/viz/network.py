"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import math

import matplotlib.pyplot as plt
import networkx as nx


RC_PARAMS = {'axes.axisbelow': True,
             'axes.edgecolor': '.8',
             'axes.facecolor': 'white',
             'axes.grid': False,
             'axes.labelcolor': '.15',
             'axes.linewidth': 1.0,
             'figure.facecolor': 'white',
             'font.family': ['sans-serif'],
             'font.sans-serif': ['Arial', 'Liberation Sans', 'sans-serif'],
             'grid.color': '.8', 'grid.linestyle': '-',
             'image.cmap': 'Greys',
             'legend.frameon': False,
             'legend.numpoints': 1, 'legend.scatterpoints': 1,
             'lines.solid_capstyle': 'round',
             'text.color': '.15',
             'xtick.color': '.15', 'xtick.direction': 'out',
             'xtick.major.size': 0.0, 'xtick.minor.size': 0.0,
             'ytick.color': '.15', 'ytick.direction': 'out',
             'ytick.major.size': 0.0, 'ytick.minor.size': 0.0}


def draw_semantic_network(graph, node_weights=None, draw_nodes=False,
                          save=False):
    """
    Draw a semantic network with nodes representing either terms or sentences,
    edges representing coocurrence or similarity, and positions given by a force-
    directed layout.

    Args:
        graph (:class:`networkx.Graph <networkx.Graph>`):
        node_weights (dict): mapping of node: weight, used to size node labels
            (and, optionally, node circles) according to their weight
        draw_nodes (bool): if True, circles are drawn under the node labels
        save (str): give the full /path/to/fname on disk to save figure (optional)

    Returns:
        ``matplotlib.axes.Axes.axis``: axis on which network plot is drawn
    """
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=(12, 12))

        pos = nx.layout.spring_layout(graph, k=2.5/math.sqrt(len(graph.nodes())))
        _ = nx.draw_networkx_edges(graph, ax=ax, alpha=0.1, pos=pos, arrows=False)

        if node_weights is None:
            if draw_nodes is True:
                _ = nx.draw_networkx_nodes(graph, ax=ax, alpha=0.25, pos=pos, linewidths=0.5)
            _ = nx.draw_networkx_labels(graph, pos, ax=ax, font_size=12,
                                        font_color='black', font_family='sans-serif')
        else:
            max_node_weight = max(node_weights.values())
            if draw_nodes is True:
                node_sizes = [600 * pow(node_weights[node]/max_node_weight, 0.75)
                              for node in graph.nodes()]
                _ = nx.draw_networkx_nodes(graph, ax=ax, pos=pos,
                                           node_size=node_sizes,
                                           alpha=0.25, linewidths=0.5)
            for node, weight in node_weights.items():
                _ = nx.draw_networkx_labels(graph, pos, labels={node: node}, ax=ax,
                                            font_color='black', font_family='sans-serif',
                                            font_size=18 * pow(weight/max_node_weight, 0.15))

        ax.set_frame_on(False)
        ax.set_xticklabels(['' for _ in range(len(ax.get_xticklabels()))])
        ax.set_yticklabels(['' for _ in range(len(ax.get_yticklabels()))])

    if save:
        fig.savefig(save, bbox_inches='tight', dpi=100)

    return ax
