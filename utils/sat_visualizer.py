import itertools
import math

import matplotlib.pyplot as plt
import networkx as nx

from data.CNFGen import SAT_3


def draw_interaction_graph(var_count: int, clauses: list):
    """ Implements visualization of interactions graphs according to http://www.carstensinz.de/papers/SAT-2005.pdf .
    Vertices are variables and edge is added between two variables if they
    are in the same clause. Darker edge means more common places.
    """
    graph = nx.Graph()
    graph.add_nodes_from([(x, {"color": "red"}) for x in range(var_count)])

    for clause in clauses:
        for u, v in itertools.combinations(clause, 2):
            v_p = abs(v) - 1
            u_p = abs(u) - 1

            if graph.has_edge(v_p, u_p):
                graph[v_p][u_p]["count"] += 1
            else:
                graph.add_edge(v_p, u_p, count=1)

    edges = graph.edges
    node_color = ["green" for _ in graph]
    edge_width = [math.log10(graph[u][v]['count']) for u, v in edges]

    options = {
        "edgelist": edges,
        "edge_color": edge_width,
        "node_color": node_color,
        "node_size": 20,
        "width": 1,
        "edge_cmap": plt.cm.Greys
    }

    pos = nx.spring_layout(graph, k=10 / math.sqrt(var_count))
    nx.draw(graph, pos, **options)
    plt.show()


def draw_factor_graph(var_count: int, clauses: list):
    """ Draws factor graph of SAT clause.
    Red edges represent negation. Blue edges represents interpretation as is.
    Cyan nodes represents clauses and green nodes represents variables.
    """
    clauses_count = len(clauses)
    graph = nx.Graph()
    graph.add_nodes_from([(x, {"color": "green"}) for x in range(var_count)])
    graph.add_nodes_from([(x, {"color": "black"}) for x in range(var_count, var_count + clauses_count, 1)])

    edges = [(abs(l - 1), idx, "b" if l > 0 else "r") for idx, c in enumerate(clauses, var_count) for l in c]
    graph.add_weighted_edges_from(edges, "color")

    edges = graph.edges
    edge_color = [graph[u][v]['color'] for u, v in edges]
    node_color = ["cyan" if node < var_count else "green" for node in graph]

    options = {
        "edge_color": edge_color,
        "node_color": node_color,
        "node_size": 20,
    }

    pos = nx.spring_layout(graph, k=10 / math.sqrt(clauses_count + var_count))
    nx.draw(graph, pos, **options)
    plt.show()


def draw_resolution_graph(clauses: list):
    """ Draws resolution graph.
    Resolution graph visualizes the structure of clause dependencies.
    Vertices are connected by edge if they have one (or more)
    literals of a different logical value.
    """
    clauses_count = len(clauses)
    graph = nx.Graph()
    graph.add_nodes_from([x for x in range(clauses_count)])

    set_clauses = [set(c) for c in clauses]

    for idx_1, c1 in enumerate(set_clauses):
        c1_inverse = {-x for x in c1}

        for idx_2, c2 in enumerate(set_clauses):
            has_common = c1_inverse.intersection(c2)

            if has_common:
                if graph.has_edge(idx_1, idx_2):
                    graph[idx_1][idx_2]["count"] += 1
                else:
                    graph.add_edge(idx_1, idx_2, count=1)

    edges = graph.edges
    node_color = ["purple" for _ in graph]
    edge_colors = ["grey" for _ in edges]

    options = {
        "edgelist": edges,
        "edge_color": edge_colors,
        "node_color": node_color,
        "node_size": 20,
        "width": 1,
    }

    pos = nx.spring_layout(graph, k=10 / math.sqrt(clauses_count))
    nx.draw(graph, pos, **options)
    plt.show()


def main():
    dataset = SAT_3("/tmp")
    var_count, clauses = [x for x in itertools.islice(dataset.train_generator(), 5)][4]
    print(clauses)

    print("Var count:", var_count)
    print("Max lit: ", max([l for c in clauses for l in c]))
    print("Min lit:", min([l for c in clauses for l in c]))

    print("Min lits in single clause: ", min([len(c) for c in clauses]))
    print("Max lits in single clause: ", max([len(c) for c in clauses]))
    draw_interaction_graph(var_count, clauses)
    draw_factor_graph(var_count, clauses)
    draw_resolution_graph(clauses)


if __name__ == '__main__':
    main()
