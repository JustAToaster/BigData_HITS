import networkx as nx
import matplotlib.pyplot as plt

def draw_hubs_graph(name, edges_list, nodes_dict, hubs_dict):
    figure = plt.gcf()
    figure.set_size_inches(120, 120)
    G_hubs = nx.DiGraph()
    G_hubs.add_edges_from(edges_list)
    G_hubs.add_nodes_from(hubs_dict.keys())
    G_hubs.add_nodes_from(nodes_dict.keys())

    # Color hub nodes with red, other nodes with grey
    node_colors_hubs = ['red' if node in hubs_dict else 'grey' for node in G_hubs.nodes()]

    # Make node size proportional to hub score if in the top 50, else use a fixed 500 size
    node_sizes_hubs = [hubs_dict[node] * 10000 if node in hubs_dict else 500 for node in G_hubs.nodes()]

    pos = nx.spring_layout(G_hubs)
    nx.draw_networkx_nodes(G_hubs, pos, cmap=plt.get_cmap('jet'), node_color=node_colors_hubs, node_size=node_sizes_hubs)
    nx.draw_networkx_labels(G_hubs, pos)
    nx.draw_networkx_edges(G_hubs, pos)

    figure.savefig("../outputs/graph_" + name + "_hub.png", format="PNG", dpi=100)
    plt.clf()

def draw_authorities_graph(name, edges_list, nodes_dict, auths_dict):
    figure = plt.gcf()
    figure.set_size_inches(120, 120)
    G_auths = nx.DiGraph()
    G_auths.add_edges_from(edges_list)
    G_auths.add_nodes_from(auths_dict.keys())
    G_auths.add_nodes_from(nodes_dict.keys())

    # Color authority nodes with blue, other nodes with grey
    node_colors_auths = ['blue' if node in auths_dict else 'grey' for node in G_auths.nodes()]

    # Make node size proportional to authority score if in the top 50, else use a fixed 500 size
    node_sizes_auths = [auths_dict[node] * 10000 if node in auths_dict else 500 for node in G_auths.nodes()]

    pos = nx.spring_layout(G_auths)
    nx.draw_networkx_nodes(G_auths, pos, cmap=plt.get_cmap('jet'), node_color=node_colors_auths, node_size=node_sizes_auths)
    nx.draw_networkx_labels(G_auths, pos)
    nx.draw_networkx_edges(G_auths, pos)

    figure.savefig("../outputs/graph_" + name + "_authorities.png", format="PNG", dpi=100)
    plt.clf()

def draw_hubs_and_authorities_graph(name, edges_list, nodes_dict, hubs_dict, auths_dict):
    figure = plt.gcf()
    figure.set_size_inches(120, 120)
    G = nx.DiGraph()
    G.add_edges_from(edges_list)
    G.add_nodes_from(auths_dict.keys())
    G.add_nodes_from(hubs_dict.keys())
    G.add_nodes_from(nodes_dict.keys())

    # Color hub nodes in red, authority nodes in blue, nodes that are both hub and authorities in purple and other nodes in grey
    node_colors = ['purple' if node in hubs_dict and node in auths_dict \
        else 'red' if node in hubs_dict \
        else 'blue' if node in auths_dict \
        else 'grey' for node in G.nodes()]

    # Make node size proportional to authority score if in the top 50, else use a fixed 500 size
    node_sizes = [hubs_dict[node] * 10000 if node in hubs_dict \
        else auths_dict[node] * 10000 if node in auths_dict \
        else 500 for node in G.nodes()]

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)

    figure.savefig("../outputs/graph_" + name + "_HubAndAuthorities.png", format="PNG", dpi=100)

    plt.close()

def draw_graphs(name, edges_list, nodes_dict, hubs_dict, auths_dict):
    
    # Graph visualization: hubs
    print("Drawing hubs graph...")
    draw_hubs_graph(name, edges_list, nodes_dict, hubs_dict)
    
    # Graph visualization: authorities
    print("Drawing authorities graph...")
    draw_authorities_graph(name, edges_list, nodes_dict, auths_dict)

    # Graph visualization: both hub and authorities
    print("Drawing hub and authorities graph...")
    draw_hubs_and_authorities_graph(name, edges_list, nodes_dict, hubs_dict, auths_dict)
