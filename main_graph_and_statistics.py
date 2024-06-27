import csv
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
import networkx as nx
import numpy as np
from collections import defaultdict
import random
# function to find bridges with DFS
def find_bridges_with_info(graph):
    bridges = set()  # we use a set to store unique bridges
    components_after_removal = []

    def dfs(node, parent, visited, disc, low):
        nonlocal time
        visited[node] = True
        disc[node] = time
        low[node] = time
        time += 1

        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, node, visited, disc, low)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    # adding the bridge as a tuple (sorted for consistency)
                    bridges.add(tuple(sorted((node, neighbor))))
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])

    def count_components(graph):
        visited = {node: False for node in graph}
        components = 0
        for node in graph:
            if not visited[node]:
                components += 1
                dfs(node, None, visited, {}, {})
        return components

    time = 0
    visited = {node: False for node in graph}
    disc = {node: float('inf') for node in graph}
    low = {node: float('inf') for node in graph}

    total_nodes = len(graph)

    for node in graph:
        if not visited[node]:
            components_after_removal.append(count_components(graph))
            dfs(node, None, visited, disc, low)

    return list(bridges), total_nodes, components_after_removal

# setting array and first values for flags
researcher_array = []
# creating an undirected graph
G = nx.Graph()
i = 0
z = 0
flag_number = 1
fnallloop = 0

# opening our csv to read it
with open('./complete_data.csv', 'r', encoding='utf-8') as f:
    data = csv.reader(f)
    headers = next(data) #first row are the headers

    for row in tqdm(data):
        if row[0] == "flag":
            if flag_number==0:
                flag_number= flag_number +1
                print("if flag")
                i = 0
                continue
            if flag_number==1:
                flag_number = 1
                z=1
                fnallloop=1
                print(project)
                print("The original list : " + str(researcher_array))
                # all possible pairs in List
                # using list comprehension + enumerate()
                res = [(a, b) for idx, a in enumerate(researcher_array) for b in researcher_array[idx + 1:]]
                # printing result
                print("All possible pairs : " + str(res))
                if len(researcher_array) == 1:
                    # if there's only one researcher, we add them as a node
                    G.add_node(researcher_array[0])
                else:
                    for x in range(len(res)):
                         print (res[x])
                    var1, var2 = zip(*res)
                    # printing the two new researchers
                    print(var1)
                    print(var2)
                    for x in range(len(res)):
                        # adding the nodes and the edges between those two
                        G.add_node(var1[x])
                        G.add_node(var2[x])
                        G.add_edge(var1[x], var2[x], label=project)
            continue
        # creating the whole array
        researcher_array.append(row[0])
        project=row[1]
        i= i + 1
        print(row[0])
        print(researcher_array)
        if z == 1:
            researcher_array = []
            researcher_array.append(row[0])
            z=0
    print("total edges number are")
    print(len(G.edges()))
    print("total nodes number are")
    print(len(G.nodes()))
    degreeView = G.degree()

    # calculating the degree of each node
    node_degrees = dict(G.degree())

    # sorting the nodes based on their degree in descending order
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

    # printing the top ten nodes and their degrees
    print("Top 10 nodes with highest degree:")
    for node, degree in sorted_nodes[:10]:
        print("Ερευνητής:", node, "| Βαθμός:", degree)

    # calculating the PageRank of each node
    pagerank = nx.pagerank(G)

    # sorting the PageRank dictionary by values in descending order
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    # taking the top 50 nodes based on PageRank
    top_50_pagerank = sorted_pagerank[:50]

    # printing the PageRank of each of the top 50 nodes with up to four decimal places
    for node, rank in top_50_pagerank:
        print("Ερευνητής:", node, "| PageRank: {:.4f}".format(rank))

    # calculating clustering coefficient for the entire graph
    clustering_coefficient = nx.average_clustering(G)
    print("Clustering Coefficient for the entire graph:", clustering_coefficient)
    # getting node degrees
    degrees = dict(G.degree())
    # getting the clustering coefficients for all nodes
    clustering_coefficients = nx.clustering(G)

    clustering_coefficient = nx.average_clustering(G)
    print("Clustering Coefficient for the entire graph:", clustering_coefficient)

    # printing the clustering coefficient for each node
    print("Clustering coefficients for each node:")
    for node, coefficient in clustering_coefficients.items():
        print(f"Node: {node}, Coefficient: {coefficient}")


    # getting the clustering coefficients for all nodes
    clustering_coefficients = nx.clustering(G)

    # total number of coefficients
    total_coefficients = len(clustering_coefficients)
    print("Total number of coefficients:", total_coefficients)

    # top 10 coefficients
    top_10 = sorted(clustering_coefficients.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 coefficients:")
    for node, coefficient in top_10:
        print(f"Node: {node}, Coefficient: {coefficient}")

    # counting of nodes with coefficient 1 and 0
    count_coefficient_1 = sum(1 for coeff in clustering_coefficients.values() if coeff == 1)
    count_coefficient_0 = sum(1 for coeff in clustering_coefficients.values() if coeff == 0)

    print("Number of nodes with coefficient 1:", count_coefficient_1)
    print("Number of nodes with coefficient 0:", count_coefficient_0)


    # lowest 10 coefficients
    lowest_10 = sorted(clustering_coefficients.items(), key=lambda x: x[1])[:10]
    print("Lowest 10 coefficients:")
    for node, coefficient in lowest_10:
        print(f"Node: {node}, Coefficient: {coefficient}")

    # filtering out nodes with degree 0
    filtered_degrees = {node: degree for node, degree in degrees.items() if degree != 0}
    nodes = list(filtered_degrees.keys())
    node_degrees = list(filtered_degrees.values())

    # sorting nodes based on degree in descending order
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)


    def check_friendship_paradox(graph):
        # calculating node degrees
        degrees = dict(graph.degree())

        # computing sum of degrees
        total_degree = sum(degrees.values())

        # computing average degree
        average_degree = total_degree / len(degrees)

        # identifying nodes experiencing the friendship paradox
        paradox_nodes = [node for node, degree in degrees.items() if degree < average_degree]
        # printing information
        print("Total sum of degrees:", total_degree)
        print("Number of nodes in the graph:", len(degrees))
        print("Average degree:", average_degree)

        # returning True if any node experiences the paradox, False otherwise, and the number of paradox nodes
        return len(paradox_nodes) > 0, len(paradox_nodes)


    friendship_paradox_holds, paradox_nodes_count = check_friendship_paradox(G)
    print("Does the friendship paradox hold in the graph?", friendship_paradox_holds)
    print("Number of nodes experiencing the paradox:", paradox_nodes_count)

    density = nx.density(G)
    # calculating degree centrality for all nodes
    degree_centrality = nx.degree_centrality(G)

    # finding the node with the highest degree centrality
    most_central_node = max(degree_centrality, key=degree_centrality.get)
    most_central_degree_centrality = degree_centrality[most_central_node]

    # finding the node with the lowest degree centrality
    least_central_node = min(degree_centrality, key=degree_centrality.get)
    least_central_degree_centrality = degree_centrality[least_central_node]

    #finding the communities in the graph
    communities = nx.algorithms.community.greedy_modularity_communities(G)

    # initializing a defaultdict to store the nodes for each community
    community_nodes = defaultdict(list)

    # collecting nodes for each community
    for idx, community in enumerate(communities):
        for node in community:
            community_nodes[idx].append(node)

    # printing the number of elements in each community
    for community_id, nodes in community_nodes.items():
        print(f"Community {community_id}: {len(nodes)} elements")
        print(", ".join(map(str, nodes)))
        print()  # Add an empty line between communities for clarity


    # sorting the degree centrality dictionary by centrality values in descending order
    sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # printing the top 10 nodes with the highest degree centrality
    for node, degree in sorted_degree_centrality[:10]:
        # limiting floating-point numbers to three decimal places
        degree = "{:.3f}".format(degree)
        print(f"{node}: Degree Centrality = {degree}")



    print("Least central node:", least_central_node)
    print("Degree centrality for the least central node:", least_central_degree_centrality)

    print("Graph Density:", density)
    # creating a subgraph with only connected nodes
    G_undirected = G.to_undirected()

    connected_nodes = max(nx.connected_components(G_undirected), key=len)
    subgraph = G_undirected.subgraph(connected_nodes)

    # counting the number of nodes and edges in the subgraph
    num_nodes_subgraph = subgraph.number_of_nodes()
    num_edges_subgraph = subgraph.number_of_edges()

    print("Number of nodes in the subgraph:", num_nodes_subgraph)
    print("Number of edges in the subgraph:", num_edges_subgraph)
    transitivity_value = nx.transitivity(G)
    print("total edges number are of subgraph")
    print(len(subgraph.edges()))
    print("total nodes number are of subgraph")
    print(len(subgraph.nodes()))
    # finding all cliques in the graph
    cliques = list(nx.find_cliques(G))

    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]

    # Find the index of the largest clique
    max_clique_index = clique_sizes.index(max(clique_sizes))

    # Print the number of cliques
    num_cliques = len(cliques)
    print("Number of cliques in the graph:", num_cliques)



    # # computing the closeness centrality
    # closeness_centrality = nx.closeness_centrality(G)
    # # Sort the closeness centrality dictionary by centrality values in descending order
    # sorted_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
    #
    # # Print the top 10 nodes with the highest closeness centrality
    # for node, closeness in sorted_closeness_centrality[:10]:
    #     # we limit floating-point numbers to three decimal places
    #     closeness = "{:.3f}".format(closeness)
    #     print(f"{node}: Closeness Centrality = {closeness}")
    #
    # # computing the betweenness centrality
    # betweenness_centrality = nx.betweenness_centrality(G)
    # # sorting the betweenness centrality dictionary by centrality values in descending order
    # sorted_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    #
    # # printing  the top 10 nodes with the highest betweenness centrality
    # for node, betweenness in sorted_betweenness_centrality[:10]:
    #     # we limit floating-point numbers to three decimal places
    #     betweenness = "{:.3f}".format(betweenness)
    #     print(f"{node}: Betweenness Centrality = {betweenness}")
    #
    # # computing eigenvector centrality with a higher maximum number of iterations
    # eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)  # Increase max_iter value
    #
    # # sorting the eigenvector centrality dictionary by centrality values in descending order
    # sorted_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    #
    # # printing the top 10 nodes with the highest eigenvector centrality
    # for node, eigenvector in sorted_eigenvector_centrality[:10]:
    #     # we limit floating-point numbers to three decimal places
    #     eigenvector = "{:.3f}".format(eigenvector)
    #     print(f"{node}: Eigenvector Centrality = {eigenvector}")


    # maximum of nodes
    num_nodes = nx.number_of_nodes(G)

    # number of edges
    num_edges = nx.number_of_edges(G)

    # removing the self-loops for an extra step of consistency
    G.remove_edges_from(nx.selfloop_edges(G))




    # # show the complete graph with random colors with plt show
    # colors = ['blue', 'red', 'yellow', 'green', 'grey', 'brown', 'orange', 'purple']
    # node_colors = [random.choice(colors) for _ in G.nodes]
    # # plotting the graph
    # plt.figure(figsize=(12, 12))  # You can adjust the figure size as needed
    # # we position the nodes using spring layout
    # pos = nx.spring_layout(G)  # Default spring layout
    # # then we draw the graph with random node colors and without labels
    # nx.draw(G, pos, node_size=10, node_color=node_colors, edge_color='gray', with_labels=False)
    #
    # # displaying the plot
    # plt.show()

    # counting the number of triangles each node is involved in
    number_of_triangles = sum(nx.triangles(G).values()) / 3
    print("Number of triangles", number_of_triangles)

    # finding all connected components of size 3
    three_node_components = [component for component in nx.connected_components(G) if len(component) == 3]

    # printing the closed triangles
    print("Closed triangles in the graph:")
    for component in three_node_components:
        print(component)

    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]

    # finding the size of the largest clique
    max_clique_size = max(clique_sizes)

    # finding all cliques of maximum size
    max_cliques = [clique for clique in cliques if len(clique) == max_clique_size]



    #calling the function of bridges for the subgraph
    bridges, total_nodes, components_after_removal = find_bridges_with_info(G)

    print("Bridges in the graph:")
    print(bridges)


    # printing all cliques in the graph
    print("Cliques in the graph:")
    for i, clique in enumerate(cliques):
        print(f"Clique {i + 1}: {clique}")

    # printing the largest cliques
    print(f"Largest cliques in the graph (size {max_clique_size}):")
    for i, clique in enumerate(max_cliques):
        print(f"Max Clique {i + 1}: {clique}")

    # printing the transitivity value
    print(f"Transitivity: {transitivity_value}")

    # calcuating the assortativity with respect to degree
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    print(f"degree_assortativity: {degree_assortativity}")
    # Compute the average shortest path length for the subgraph
    # Check if the subgraph is connected
    # if nx.is_connected(subgraph):
    #     # Compute the average shortest path length for the subgraph
    #     avg_shortest_path_length_subgraph = nx.average_shortest_path_length(subgraph)
    #     print("Average Shortest Path Length for the subgraph:", avg_shortest_path_length_subgraph)
    # else:
    #     print("Graph is not connected, cannot compute average shortest path length.")

    # displaying the five nodes with the most edges
    degree_counts = Counter(dict(G.degree()))
    most_connected_nodes = degree_counts.most_common(5)
    print("Nodes with the most edges:")
    for node, degree in most_connected_nodes:
        print("Node:", node, "- Degree:", degree)
    # displaying the five nodes with the most edges
    print("Nodes with the most edges:")
    for node in sorted_nodes[:5]:
        print(node, "- Degree:", degrees[node])

    # sorting nodes based on degree in ascending order
    sorted_nodes = sorted(degrees, key=degrees.get)

    # calculating clustering coefficient for the entire graph
    # clustering_coefficient = nx.average_clustering(subgraph)

    # displaying the clustering coefficient
    # print("Clustering Coefficient for the entire graph:", clustering_coefficient)
    # calculating the  average shortest path length for the entire graph
    # computing shortest path distances between all pairs of nodes in the subgraph

    # finding all connected components
    connected_components = list(nx.connected_components(G))

    # sorting the connected components by length in descending order
    sorted_components = sorted(connected_components, key=len, reverse=True)

    # sorting connected components by length in ascending order
    sorted_components_min = sorted(connected_components, key=len)
    # checking if there are at least two connected components
    if len(sorted_components) >= 2:
        # storing the length of the first and second largest connected components
        first_largest_length = len(sorted_components[0])
        second_largest_length = len(sorted_components[1])

        print("Number of nodes in the first largest connected component:", first_largest_length)
        print("Number of nodes in the second largest connected component:", second_largest_length)
    else:
        print("There are not enough connected components.")

    # printing information about some of the smallest components in our graph
    num_smallest_components = min(7, len(sorted_components)) if len(sorted_components) >= 2 else 2
    for i in range(num_smallest_components):
        component = sorted_components_min[i]
        component_length = len(component)
        print(f"Connected Component {i + 1} (Size: {component_length}):")
        print(component)
        print()  # an empty line to read it better


    nodes = list(filtered_degrees.keys())
    # we create the scattering plot
    plt.figure(figsize=(10, 6))
    plt.scatter(nodes, node_degrees, color='blue', alpha=0.5)
    plt.title('Scatter plot Ερευνητών')
    plt.xlabel('Ερευνητές')
    plt.ylabel('Βαθμός')
    plt.xticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # we calculate degree distribution
    degree_sequence = sorted([d for n, d in G.degree()])

    # we adjust the degree distribution as a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2), color='b', edgecolor='black', alpha=0.7)
    plt.title('Κατανομή βαθμού των κόμβων Γράφημα Α')
    plt.xlabel('Ακμή')
    plt.ylabel('Συχνότητα')
    plt.grid(True)
    plt.show()



    # calculating the degree distribution
    degrees = [degree for node, degree in G.degree()]
    degree_counts = np.bincount(degrees)
    k = np.arange(len(degree_counts))

    # number of nodes and edges
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # propability of edge creation for the ER graph
    p = 2 * m / (n * (n - 1))
    # generating an Erdős–Rényi graph with the same number of nodes and similar average degree
    ER_graph = nx.erdos_renyi_graph(n, p)

    # calculating the degree distribution for the ER graph
    ER_degrees = [degree for node, degree in ER_graph.degree()]
    ER_degree_counts = np.bincount(ER_degrees)
    ER_k = np.arange(len(ER_degree_counts))

    # average degree for BA graph
    k_avg = int(np.mean([degree for node, degree in G.degree()]) / 2)

    # generating a Barabási–Albert graph with the same number of nodes and similar average degree
    BA_graph = nx.barabasi_albert_graph(n, k_avg)

    # calculating the degree distribution for the BA graph
    BA_degrees = [degree for node, degree in BA_graph.degree()]
    BA_degree_counts = np.bincount(BA_degrees)
    BA_k = np.arange(len(BA_degree_counts))

    # plotting both degree distributions in the same figure
    plt.figure(figsize=(10, 6))

    # plotting our  graph degree distribution
    plt.bar(k, degree_counts, width=0.4, color='blue', alpha=0.6, edgecolor='black', label='Γράφημα Α')



    # plotting BA graph degree distribution
    plt.bar(BA_k + 1.0, BA_degree_counts, width=0.4, color='green', alpha=0.6, edgecolor='black',
            label='Barabasi Albert Γράφημα')

    # we label the axes and title
    plt.xlabel('Βαθμός')
    plt.ylabel('Συχνότητα')
    plt.title('Σύγκριση κατανομής βαθμών')

    # we add the legend
    plt.legend()

    # shoing the plot
    plt.show()

    # plotting both degree distributions in the same figure
    plt.figure(figsize=(10, 6))

    plt.bar(k, degree_counts, width=0.4, color='blue', alpha=0.6, edgecolor='black', label='Γράφημα Α')
    plt.bar(ER_k + 0.5, ER_degree_counts, width=0.4, color='red', alpha=0.6, edgecolor='black', label='Erdos Renyi Γράφημα')

    plt.xlabel('Βαθμός')
    plt.ylabel('Συχνότητα')
    plt.title('Σύγκριση κατανομή βαθμών')
    plt.legend()

    plt.show()


    # we calculate the density
    BA_density = nx.density(BA_graph)
    ER_density = nx.density(ER_graph)

    print(f"Density of BA graph: {BA_density}")
    print(f"Density of ER graph: {ER_density}")

    # then the average clustering coefficient
    BA_clustering_coeff = nx.average_clustering(BA_graph)
    ER_clustering_coeff = nx.average_clustering(ER_graph)

    print(f"Clustering coefficient of BA graph: {BA_clustering_coeff}")
    print(f"Clustering coefficient of ER graph: {ER_clustering_coeff}")

    # we count frequencies of each degree
    degree_counts = {}
    for degree in degree_sequence:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    # sorting degree counts by degree
    sorted_degrees = sorted(degree_counts.keys())
    sorted_counts = [degree_counts[degree] for degree in sorted_degrees]

    # plotting degree distribution as lines
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_degrees, sorted_counts, color='b', marker='o', linestyle='-')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    node_least_edges = min(G.nodes(), key=lambda x: G.degree(x))
    # finding  the node with the most edges and the one with the least
    node_most_edges = max(G.nodes(), key=lambda x: G.degree(x))
    print("Node with the least edges:", node_least_edges)
    print("Node with the most edges:", node_most_edges)
    degree_counts = Counter(dict(degreeView))
    max_degree_node = degree_counts.most_common(1)
    print(max_degree_node)
f.close()
