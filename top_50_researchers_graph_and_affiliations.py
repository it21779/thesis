import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import community
from collections import defaultdict


# function to find bridges with DFS
def find_bridges(graph):
    bridges = []

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
                    bridges.append((node, neighbor))
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])

    time = 0
    visited = {node: False for node in graph}
    disc = {node: float('inf') for node in graph}
    low = {node: float('inf') for node in graph}

    for node in graph:
        if not visited[node]:
            dfs(node, None, visited, disc, low)

    return bridges

# creating an empty graph
graph = nx.Graph()
# creating an empty graph for affiliations
affiliations_graph = nx.Graph()

# setting to store first names
first_names = set()

# list to store project nodes
project_nodes = []

# dictionary to store projects and associated names
projects = {}
name_affiliation_project_count = {}

# dictionary to store projects per affiliation
affiliation_projects_count = {}

# list to store affiliations
affiliations = []
affiliation_names = []

# an empty dictionary to store memory of each name and its affiliation
name_affiliation_memory = {}

# read names from text file
with open('top_50_names.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        data_line = line.strip().split(',')
        name = data_line[0]
        first_names.add(name)

        # we add the name node to the main graph with affiliation information
        graph.add_node(name, affiliation=[data_line[-1].strip()])  # Adding affiliation info here
        print(f"Creating node: {name}")
        # then we extract affiliation name from the end of the row
        affiliation_name = data_line[-1].strip()
        # we add the affiliation name to the list
        affiliation_names.append(affiliation_name)
        # storing the rest of the data in different variables
        variables = data_line[1:-1]  # Exclude the last element which is the affiliation
        # storing the memory of the name along with its affiliation
        name_affiliation_memory[name] = affiliation_name

        # we update the project count for the name
        if name in name_affiliation_project_count:
            name_affiliation_project_count[name]['projects'] += len(variables)
        else:
            name_affiliation_project_count[name] = {'affiliation': affiliation_name, 'projects': len(variables)}

        # we track projects associated with each name and affiliation
        for variable in variables:
            if variable in projects:
                projects[variable].append(name)
            else:
                projects[variable] = [name]

            # then we update affiliation projects count
            if affiliation_name in affiliation_projects_count:
                if variable not in affiliation_projects_count[affiliation_name]:
                    affiliation_projects_count[affiliation_name].append(variable)
            else:
                affiliation_projects_count[affiliation_name] = [variable]

    # printing some info like projects for each line
    for line in lines:
        data_line = line.strip().split(',')
        projects_line = data_line[1:-1]  # Extract projects from the line
        print("Projects for this line:", projects_line)

    # then names, affiliations, and project counts
    for name, info in name_affiliation_project_count.items():
        print(f"Name: {name}, Affiliation: {info['affiliation']}, Projects: {info['projects']}")

    # a counter for projects per affiliation
    print("\nProjects per affiliation:")
    for affiliation, project_list in affiliation_projects_count.items():
        print(f"Affiliation: {affiliation}, Unique Projects: {len(project_list)}")

    # printing all names and their affiliations
    for name, affiliation in name_affiliation_memory.items():
        print(f"Name: {name}, Affiliation: {affiliation}")
    # code to find the affiliation with the least amount of projects
    min_projects_affiliation = min(affiliation_projects_count, key=lambda k: len(affiliation_projects_count[k]))

    # then printing the details of the affiliation with the least amount of projects
    print(f"\nAffiliation with the least amount of projects: {min_projects_affiliation}")
    print(f"Projects for {min_projects_affiliation}:")
    for project in affiliation_projects_count[min_projects_affiliation]:
        print(project)

    # dictionary to store the count of members for each university
    university_member_counts = {}

    # counting the members for each university
    for affiliation in set(name_affiliation_memory.values()):
        university_member_counts[affiliation] = list(name_affiliation_memory.values()).count(affiliation)

    # Sort the dictionary by the count of members in descending order
    sorted_university_member_counts = dict(sorted(university_member_counts.items(), key=lambda item: item[1], reverse=True))

    # printing the ranking of universities based on the number of members
    print("Ranking of Affiliations by Number of Members:")
    for index, (university, count) in enumerate(sorted_university_member_counts.items(), start=1):
        print(f"{index}. {university}: {count} members")

    universities = list(sorted_university_member_counts.keys())
    member_counts = list(sorted_university_member_counts.values())

    # we create the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(universities, member_counts, color='skyblue')
    plt.xlabel('Number of Members')
    plt.ylabel('Affiliation')
    plt.title('Ranking of Universities by Number of Members')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # displaying the barchart
    plt.show()

    # dictionary to store names of researchers for each university
    university_researchers = {affiliation: [] for affiliation in sorted_university_member_counts.keys()}

    # we add to the dictionary  names of researchers for each university
    for name, affiliation in name_affiliation_memory.items():
        university_researchers[affiliation].append(name)

    # printing the names of researchers next to each university
    print("Researchers by University:")
    for university, researchers in university_researchers.items():
        print(f"{university}:")
        for researcher in researchers[:50]:  # it can have up to 50 researchers per university
            print(f"- {researcher}")
        print()  # Add a blank line between universities

    # we find the affiliation with the most members
    max_members_affiliation = max(name_affiliation_memory.values(), key=lambda x: list(name_affiliation_memory.values()).count(x))

    # printing details of the affiliation with the most members
    print(f"\nAffiliation with the most members: {max_members_affiliation}")

    # finding the affiliation with the most projects
    max_projects_affiliation = max(affiliation_projects_count, key=lambda k: len(affiliation_projects_count[k]))

    # printing details of the affiliation with the most projects
    print(f"\nAffiliation with the most projects: {max_projects_affiliation}")
    print(f"Projects for {max_projects_affiliation}:")
    for project in affiliation_projects_count[max_projects_affiliation]:
        print(project)


    # our data
    affiliations = list(affiliation_projects_count.keys())
    projects_count = [len(projects) for projects in affiliation_projects_count.values()]

    # we create the bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(affiliations, projects_count, color='skyblue')
    plt.xlabel('Number of Projects')
    plt.ylabel('Affiliation')
    plt.title('Number of Projects per Affiliation')

    # adjusting margins
    plt.subplots_adjust(left=0.5)

    plt.gca().invert_yaxis()
    plt.show()

    #code for main graph with top 50 researchers
    # creating edges for each project and associated names in the main graph
    for project, associated_names in projects.items():
        if len(associated_names) > 1:
            # if the project has multiple parents, connect all associated names
            for i in range(len(associated_names)):
                for j in range(i + 1, len(associated_names)):
                    graph.add_edge(associated_names[i], associated_names[j], label=project)
                    print(f"Creating edge: {associated_names[i]} -> {associated_names[j]} (project: {project})")
        elif len(associated_names) == 1:
            # if the project has only one creator, add a normal edge
            creator = associated_names[0]
            graph.add_edge(creator, project, label=project)
            print(f"Creating edge: {creator} -> {project} (project: {project})")

            # add the project node to the project_nodes list
            project_nodes.append(project)

    # counting the number of name nodes and connections
    num_name_nodes = 0
    num_connections = 0
    # since its project also counts as a node we will need to exclude them
    for node in graph.nodes:
        if node not in project_nodes:
            # we exlude project nodes when counting name nodes
            neighbors = list(graph.neighbors(node))
            if neighbors:
                num_name_nodes += 1
                num_connections += len(neighbors)

    print(f"Number of name nodes: {num_name_nodes}")
    print(f"Number of connections to other name nodes: {num_connections}")
    # calculating the number of connections to other name nodes
    num_connections = {}

    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))

        # we exclude connections to project nodes
        name_neighbors = [neighbor for neighbor in neighbors if neighbor not in project_nodes]

        num_connections[node] = len(name_neighbors)

    # we calculate the degree centrality for each individual (excluding project nodes)
    degree_centrality = {}
    for node in graph.nodes:
        if node not in project_nodes:
            neighbors = list(graph.neighbors(node))
            name_neighbors = [neighbor for neighbor in neighbors if neighbor not in project_nodes]
            degree_centrality[node] = len(name_neighbors) / (num_name_nodes)  # Exclude the node itself

    # checking for the friends paradox
    friends_paradox = True
    problematic_node = None
    problematic_statistic = None

    # finding the node with the highest degree centrality
    most_central_node = min(degree_centrality, key=degree_centrality.get)

    # printing the most central node and its degree centrality
    print("Most central node:", most_central_node)
    print("Degree centrality:", degree_centrality[most_central_node])

    for node, num_connections_value in num_connections.items():
        if node not in project_nodes:
            degree_centrality_value = degree_centrality[node]

            if degree_centrality_value > num_connections_value:
                friends_paradox = False
                problematic_node = node
                problematic_statistic = "Degree Centrality"
                break

    # the result
    if friends_paradox:
        print("The friends paradox holds in the network.")
    else:
        print(f"The friends paradox does not hold. Problematic Node: {problematic_node}, Statistic: {problematic_statistic}")

    # we create empty arrays to store node names and degrees
    node_names = []
    node_degrees = []

    # printing degree of each name excluding project nodes and project-related nodes
    for node in graph.nodes:
        if node not in project_nodes:
            neighbors = list(graph.neighbors(node))
            name_neighbors = [neighbor for neighbor in neighbors if neighbor not in project_nodes]
            degree = len(name_neighbors)
            print(f"{node}: Degree = {degree}")

            # storing node name and degree in arrays
            node_names.append(node)
            node_degrees.append(degree)
    # creating a subgraph excluding project nodes
    subgraph_nodes = [node for node in graph.nodes if node not in project_nodes]
    subgraph = graph.subgraph(subgraph_nodes)

    # we create a modifiable copy of the subgraph, we will do different calculations in this one
    subgraph_modifiable = subgraph.copy()

    # getting all edges in the subgraph
    all_edges = list(subgraph_modifiable.edges())

    # finding and removing and remove self-loops
    self_loops = [(u, v) for u, v in all_edges if u == v]
    subgraph_modifiable.remove_edges_from(self_loops)

    univ_graph = subgraph_modifiable.copy()

    # we iterate over nodes in univ_graph
    for node in univ_graph.nodes:
        # If node name exists and it has an affiliation in the memory dictionary
        if node in name_affiliation_memory:
            # Get the affiliation from the memory dictionary
            affiliation_name = name_affiliation_memory[node]
            # Update the name of the node in univ_graph with the affiliation name
            univ_graph = nx.relabel_nodes(univ_graph, {node: affiliation_name})


    #we find the communities of our original graph with the top 50 researchers
    communities = community.best_partition(subgraph_modifiable)

    # we initialise a defaultdict to store the nodes for each community
    community_nodes = defaultdict(list)

    # we append the nodes for each community
    for node, community_id in communities.items():
        community_nodes[community_id].append(node)

    # printing the number of elements in each community
    for community_id, nodes in community_nodes.items():
        print(f" flag1 Community {community_id}: {len(nodes)} elements")
        print(", ".join(nodes))
        print()  # adding an empty line between communities for clarity



    # drawing the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(subgraph_modifiable, seed=42)
    from itertools import cycle
    #different colors for the nodes
    colors = cycle(['red', 'blue', 'yellow', 'green', 'orange', 'grey'])

    # drawing nodes with colors according to their communities
    for community_id, nodes in community_nodes.items():
        color = next(colors)  # Cycle through the list of colors
        nx.draw_networkx_nodes(subgraph_modifiable, pos, nodelist=nodes, node_color=color, node_size=500)

    # drawing edges with lighter color
    nx.draw_networkx_edges(subgraph_modifiable, pos, edge_color='lightgray', connectionstyle="arc3,rad=0.1")

    # drawing labels
    nx.draw_networkx_labels(subgraph_modifiable, pos, font_size=8, font_color='black', font_family='sans-serif')

    # customising plot
    plt.title("Γράφημα με Ερευνητές και τις Συνδέσεις τους.")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    # calculating the degree of each node
    node_degrees = dict(subgraph_modifiable.degree())

    # sorting the nodes based on their degree in descending order
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

    # printing the top ten nodes and their degrees
    print("Top 10 nodes with highest degree:")
    for node, degree in sorted_nodes[:10]:
        print("Ερευνητής:", node, "| Βαθμός:", degree)


    # calculating the PageRank of each node
    pagerank = nx.pagerank(subgraph_modifiable)

    # sorting the PageRank dictionary by values in descending order
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    # printing the PageRank of each node
    for node, rank in sorted_pagerank:
        print("Ερευνητής:", node, "| PageRank: {:.4f}".format(rank))

    #we find the bridges for our entire graph
    bridges = find_bridges(subgraph_modifiable)
    print("Bridges in the subgraph:")
    print(bridges)


    # checking if the graph is connected, if it isnt we cant count some statistics
    if nx.is_connected(graph):
        # we caclulate the diameter of the graph
        diameter = nx.diameter(graph)
        # and then we print the diameter
        print(f"Graph Diameter: {diameter}")
        name_nodes = [node for node in graph.nodes if node not in project_nodes]
        global_clustering_coefficient = nx.average_clustering(graph, nodes=name_nodes)
        print(f"Global Average Clustering Coefficient: {global_clustering_coefficient}")
        # calculating local clustering coefficient for each node
        local_clustering_coefficients = nx.clustering(graph, nodes=name_nodes)
        # calculating the radius and diameter of the graph
        radius = nx.radius(subgraph)
        diameter = nx.diameter(subgraph)
        # printing the radius and diameter
        print(f"Radius of the graph: {radius}")
        print(f"Diameter of the graph: {diameter}")
        # printing local clustering coefficient for each node
        for node, clustering_coefficient in local_clustering_coefficients.items():
            if node not in project_nodes:
                print(f"{node}: Local Clustering Coefficient = {clustering_coefficient}")
    else: #thats why we count the more generic statistics here
        name_nodes = [node for node in graph.nodes if node not in project_nodes]
        global_clustering_coefficient = nx.average_clustering(graph, nodes=name_nodes)
        local_clustering_coefficients = nx.clustering(graph, nodes=name_nodes)
        print(f"Global Average Clustering Coefficient: {global_clustering_coefficient}")
        for node, clustering_coefficient in local_clustering_coefficients.items():
            # we can calculate local clustering coefficient for each node
            if node not in project_nodes:
                print(f"{node}: Local Clustering Coefficient = {clustering_coefficient}")
        print("Can not find graph diameter. The graph is not connected.")
        print("Can not find graph radius and diameter. The graph is not connected.")

    name_nodes = [node for node in graph.nodes if node not in project_nodes]
    # perform community detection using the Louvain method, only on 'name_nodes'
    partition = community.best_partition(graph.subgraph(name_nodes))

    # print communities
    for node, community_id in partition.items():
        print(f"Node {node} belongs to community {community_id}")

    # initialize a defaultdict to store the nodes for each community
    community_nodes = defaultdict(list)

    # populate the defaultdict with nodes for each community
    for node, community_id in partition.items():
        community_nodes[community_id].append(node)

    # finding the largest community
    largest_community_id = max(community_nodes, key=lambda x: len(community_nodes[x]))
    largest_community_nodes = community_nodes[largest_community_id]

    # printing the number of nodes in each community
    print("Number of Nodes in Each Community:")
    for community_id, nodes in community_nodes.items():
        print(f"  Community {community_id}: {len(nodes)} nodes")
        print(", ".join(nodes))
        print()  # adding an empty line between communities for clarity

    # getting the number of nodes in the subgraph_modifiable
    num_nodes = subgraph_modifiable.number_of_nodes()

    # getting the number of edges in the subgraph_modifiable
    num_edges = subgraph_modifiable.number_of_edges()

    # printing those two numbers
    print("Number of nodes in the subgraph_modifiable:", num_nodes)
    print("Number of edges in the subgraph_modifiable:", num_edges)

    # calculating the closeness centrality excluding project nodes
    closeness_centrality = nx.closeness_centrality(subgraph_modifiable, u=None)

    # calculating the betweenness centrality excluding project nodes
    betweenness_centrality = nx.betweenness_centrality(subgraph_modifiable)

    # printing the closeness centrality for each name node
    for node, closeness in closeness_centrality.items():
        if node not in project_nodes:
            print(f"{node}: Closeness Centrality = {closeness}")

    # sorting the closeness centrality dictionary by centrality values in descending order
    sorted_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

    # printing the top 10 nodes with the highest closeness centrality
    for node, closeness in sorted_closeness_centrality[:10]:
        # we limit the floating-point numbers to three decimal places
        closeness = "{:.3f}".format(closeness)
        print(f"{node}: Closeness Centrality = {closeness}")

    # sorting the betweenness centrality dictionary by centrality values in descending order
    sorted_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

    # printing the top 10 nodes with the highest betweenness centrality
    for node, betweenness in sorted_betweenness_centrality[:10]:
        # we limit floating-point numbers to three decimal places
        betweenness = "{:.3f}".format(betweenness)
        print(f"{node}: Betweenness Centrality = {betweenness}")



    # sorting the degree centrality dictionary by centrality values in descending order
    sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # we print the top 10 nodes with the highest degree centrality
    for node, degree in sorted_degree_centrality[:10]:
        # Limit floating-point numbers to three decimal places
        degree = "{:.3f}".format(degree)
        print(f"{node}: Degree Centrality = {degree}")


    # then we calculate the eigenvector centrality excluding project nodes
    eigenvector_centrality = nx.eigenvector_centrality(subgraph_modifiable)

    # sorting the eigenvector centrality dictionary by centrality values in descending order
    sorted_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

    # we print the top 10 nodes with the highest eigenvector centrality
    for node, eigenvector in sorted_eigenvector_centrality[:10]:
        # and showing only floating-point numbers to three decimal places
        eigenvector = "{:.3f}".format(eigenvector)
        print(f"{node}: Eigenvector Centrality = {eigenvector}")

    #we use the nx.connected_components in the graph without the project nodes
    connected_components = list(nx.connected_components(subgraph_modifiable))

    # we find the largest connected component and we print the edges and the nodes of it
    largest_component_normal_researchers = max(connected_components, key=len)

    num_nodes_largest_component = len(subgraph_modifiable.nodes())
    num_edges_largest_component = len(subgraph_modifiable.edges())
    print("Number of nodes in the largest connected component:", num_nodes_largest_component)
    print("Number of edges in the largest connected component:", num_edges_largest_component)


    # we calculate the edge betweenness centrality for all edges in the graph
    edge_betweenness = nx.edge_betweenness_centrality(subgraph_modifiable)

    # sorting the edge betweenness dictionary by centrality values in descending order
    sorted_edge_betweenness = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

    # printing the top ten edges with the highest betweenness centrality
    print("Top ten edges with the highest edge betweenness centrality:")
    for edge, centrality in sorted_edge_betweenness[:10]:
        print(f"Edge {edge}: Edge Betweenness Centrality = {centrality}")

    # we count the number of triangles each node is involved in
    number_of_triangles = sum(nx.triangles(subgraph_modifiable).values()) / 3
    print("Number of triangles",number_of_triangles)

    # computing shortest path distances between all pairs of nodes in the subgraph
    shortest_paths = dict(nx.all_pairs_shortest_path_length(subgraph))

    # printing shortest path distances
    print("Shortest Path Distances in Subgraph:")
    for node, distances in shortest_paths.items():
        print(f"Node {node}: {distances}")

    shortest_paths = dict(nx.all_pairs_shortest_path_length(subgraph))


    total_paths = 0
    total_length = 0

    # setting to keep track of already encountered shortest paths
    encountered_paths = set()

    # iterating over the shortest path distances
    for node, distances in shortest_paths.items():
        for target_node, distance in distances.items():
            # we ccheck if the reverse path has been encountered
            reverse_path = (target_node, node)
            if reverse_path not in encountered_paths:
                # we increase the count and total length
                total_paths += 1
                total_length += distance
                # adding the path to encountered paths to avoid duplicates
                encountered_paths.add((node, target_node))
    # computing the connected components of the subgraph
    connected_components = list(nx.connected_components(subgraph))

    # we initialise variables to store eccentricities of non-isolated nodes
    eccentricities = {}

    # iterating over connected components
    for component in connected_components:
        # checking if the component is non-isolated (contains more than one node)
        if len(component) > 1:
            # computing eccentricity for each node in the component
            component_eccentricities = nx.eccentricity(subgraph.subgraph(component))
            # updating eccentricities dictionary with eccentricities from the current component
            eccentricities.update(component_eccentricities)

    # checking if there are any non-isolated nodes
    if eccentricities:
        # computing diameter and radius
        diameter = max(eccentricities.values())
        radius = min(eccentricities.values())
        print("Subgraph Diameter:", diameter)
        print("Subgraph Radius:", radius)
        # we get all nodes in the subgraph
        subgraph_nodes = list(subgraph.nodes())
    else:
        print("Subgraph contains only isolated nodes. Diameter and Radius cannot be computed.")


    # we calculate the average shortest path length
    if total_paths > 0:
        average_shortest_path_length = total_length / total_paths
        print("Average Shortest Path Length in Subgraph:", average_shortest_path_length)
    else:
        print("No unique shortest paths found in the subgraph.")

    # we calculate node degrees
    degrees = dict(subgraph.degree())

    # computing the average degree
    average_degree = sum(degrees.values()) / len(degrees)

    # identifying nodes experiencing the friendship paradox
    paradox_nodes = [node for node, degree in degrees.items() if degree < average_degree]

    # we print if it stands based the numbers holding
    print("Node Degrees:", degrees)
    print("Average Degree:", average_degree)
    print("Nodes experiencing the friendship paradox:", paradox_nodes)

    if nx.is_connected(subgraph):
        # computing eccentricity for each node in the subgraph
        eccentricities = nx.eccentricity(subgraph)
        # printing eccentricity for each node in the subgraph
        print("\nEccentricity in Subgraph:")
        for node, eccentricity in eccentricities.items():
            print(f"Node {node}: Eccentricity = {eccentricity}")
    else:
        print("Subgraph is not connected. Eccentricity can not be calculated")
    # calculating transitivity for the subgraph
    transitivity_value = nx.transitivity(subgraph)
    # printing the transitivity value
    print(f"Transitivity: {transitivity_value}")
    # getting the number of nodes
    num_nodes = subgraph_modifiable.number_of_nodes()

    # getting the number of edges
    num_edges = subgraph_modifiable.number_of_edges()
    # calculating assortativity with reference to degree
    degree_assortativity = nx.degree_assortativity_coefficient(subgraph)

    # printing the degree assortativity coefficient
    print(f"Degree Assortativity: {degree_assortativity}")
    # printing the number of nodes and edges in the subgraph
    num_nodes_subgraph = subgraph_modifiable.number_of_nodes()
    num_edges_subgraph = subgraph_modifiable.number_of_edges()
    print(f"Number of nodes in subgraph: {num_nodes_subgraph}")
    print(f"Number of edges in subgraph: {num_edges_subgraph}")

    # getting the node degrees for each node
    node_degrees1 = dict(subgraph.degree())  # Example node_degrees1

    # extracting node degrees into an array
    degree_distribution = list(node_degrees1.values())

    # printing the degree distribution array
    print("Degree Distribution:")
    print(degree_distribution)

    # printing the total number of nodes
    print("Total Number of Nodes:", num_nodes)
    # Calculate the average degree
    average_degree = np.mean(degree_distribution)

    # printing the average degree
    print("Average Degree:", average_degree)

    # printing all node degrees
    print("Node Degrees:")
    for i, degree in enumerate(degree_distribution):
        print(f"Node {i+1}: Degree = {degree}")
        # calculating the sum of all node degrees
        total_degree_sum = sum(degree_distribution)
    print(f"Nnum_edges_subgraphin subgraph: {num_edges_subgraph}")
    # printing the sum of all node degrees
    print(f"Sum of all node degrees: {total_degree_sum}")
    print(f'Average Degree of Name Nodes: {average_degree:.2f}')
    # identify connected components excluding project nodes
    connected_components = [component for component in nx.connected_components(graph) if not any(node in project_nodes for node in component)]

    # printing degree of each component excluding project nodes
    for component in connected_components:
        component_degree = np.mean([graph.degree(node) for node in component if node not in project_nodes])
        print(f"Component: {component}, Degree: {component_degree:.2f}")

    # Calculate and print the average degree for name nodes
    average_degree = np.mean(degree_distribution)
    print(f'Average Degree of Name Nodes: {average_degree:.4f}')
    # computing PageRank
    pagerank_scores = nx.pagerank(subgraph)

    # printing PageRank scores for each node
    print("PageRank Scores:")
    for node, score in pagerank_scores.items():
        print(f"{node}: {score}")

    # finding all cliques in the graph
    cliques = list(nx.find_cliques(subgraph))
    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]
    # finding the index of the largest clique
    max_clique_index = clique_sizes.index(max(clique_sizes))
    # printing the number of cliques
    num_cliques = len(cliques)
    print("Number of cliques in the graph:", num_cliques)

    # finding all cliques in the graph
    cliques = list(nx.find_cliques(subgraph))

    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]

    # finding the size of the largest clique
    max_clique_size = max(clique_sizes)

    # finding all cliques of maximum size
    max_cliques = [clique for clique in cliques if len(clique) == max_clique_size]

    # printing the number of cliques
    num_cliques = len(cliques)
    print("Number of cliques in the graph:", num_cliques)

    # printing all cliques in the graph
    print("Cliques in the graph:")
    for i, clique in enumerate(cliques):
        print(f"Clique {i + 1}: {clique}")

    # printing the largest cliques
    print(f"Largest cliques in the graph (size {max_clique_size}):")
    for i, clique in enumerate(max_cliques):
        print(f"Max Clique {i + 1}: {clique}")

    # calculating density of subgraph
    density = nx.density(subgraph)

    print("Graph Density:", density)
    # plotting the degree distribution
    plt.figure(figsize=(10, 5))

    node_least_edges = min(subgraph.nodes(), key=lambda x: subgraph.degree(x))

    # finding the node with the most edges
    node_most_edges = max(subgraph.nodes(), key=lambda x: subgraph.degree(x))

    print("Node with the least edges:", node_least_edges)
    print("Node with the most edges:", node_most_edges)
    # our histogram
    plt.subplot(1, 2, 1)
    plt.hist(degree_distribution, bins='auto', edgecolor='black')
    plt.axvline(x=average_degree, color='red', linestyle='dashed', linewidth=2, label=f'Μέσος αριθμός ακμών: {average_degree:.4f}')
    plt.xlabel('Ακμή')
    plt.ylabel('Συχνότητα')
    plt.title(' Κατανομή βαθμού των Κόμβων')
    plt.legend()
    node_degrees = dict(subgraph.degree())
    # extracting node names and degrees into separate lists
    node_names = list(node_degrees.keys())
    degrees = list(node_degrees.values())

    # creating numerical indices for the nodes
    node_indices = range(len(node_names))
    # scattering the plot
    plt.subplot(1, 2, 2)
    plt.scatter(node_indices, degrees, color='blue', alpha=0.5, edgecolors='black')
    plt.xlabel('Ερευνητές')
    plt.ylabel('Βαθμοί')
    plt.xticks([])
    plt.title('Κατανομή Βαθμού ανά Κόμβο')

    plt.tight_layout()
    plt.show()

    # creating an empty graph
    affiliations2_graph = nx.Graph()

    # dictionary to store edges between affiliations for common projects
    common_projects_edges = {}

    # iterate through each affiliation and its associated projects
    for affiliation, project_list in affiliation_projects_count.items():
        # sorting the project list to ensure consistency in edge creation
        project_list.sort()
        # checking for common projects with other affiliations
        for other_affiliation, other_project_list in affiliation_projects_count.items():
            if affiliation != other_affiliation:
                # sorting the other project list to ensure consistency in edge creation
                other_project_list.sort()
                # finding common projects between the two affiliations
                common_projects = set(project_list).intersection(other_project_list)
                if common_projects:
                    # creating an edge for each common project
                    for project in common_projects:
                        if (affiliation, other_affiliation) not in common_projects_edges:
                            common_projects_edges[(affiliation, other_affiliation)] = []
                        common_projects_edges[(affiliation, other_affiliation)].append(project)

    # adding nodes to the graph
    affiliations2_graph.add_nodes_from(affiliation_projects_count.keys())



    # adding edges to the graph with weights based on the number of common projects
    for (affiliation1, affiliation2), projects in common_projects_edges.items():
         weight = len(projects)  # Weight based on the number of common projects
         affiliations2_graph.add_edge(affiliation1, affiliation2, project_count=weight)
         print("added edge between", affiliation2, affiliation1, weight)

     # drawing the graph with modified layout
    pos = nx.spring_layout(affiliations2_graph, k=0.2, iterations=20, seed=42)  # Adjust k and iterations for spreading

    # drawing nodes
    nx.draw_networkx_nodes(affiliations2_graph, pos, node_color='skyblue', node_size=500)

    # drawing edges with weights
    edge_weights = {(u, v): affiliations2_graph[u][v]['project_count'] for u, v in affiliations2_graph.edges()}
    nx.draw_networkx_edges(affiliations2_graph, pos, edgelist=affiliations2_graph.edges(), width=1.0, alpha=0.5, edge_color='k')

    # drawing edge labels
    nx.draw_networkx_edge_labels(affiliations2_graph, pos, edge_labels=edge_weights, font_color='red')

    # drawing node labels
    nx.draw_networkx_labels(affiliations2_graph, pos, font_size=7, font_family='sans-serif')

    # showing plot
    plt.title("Affiliations Graph with Common Projects")
    plt.axis('off')
    plt.show()

    # finding the bridges in our affiliation graph
    bridges = find_bridges(affiliations2_graph)
    print("Bridges in the affiliations2_graph:")
    print(bridges)

    # calculating the PageRank of each node
    pagerank = nx.pagerank(affiliations2_graph)

    # printing the PageRank of each node with three decimals
    for node, rank in pagerank.items():
        print("Affiliation:", node, "| PageRank: {:.3f}".format(rank))

    #finding the eccentricities in our affiliation graph
    eccentricities = nx.eccentricity(affiliations2_graph)

    # printing the eccentricity of each node
    print("Eccentricity of each node:")
    for node, eccentricity in eccentricities.items():
        print(f"Node {node}: Eccentricity = {eccentricity}")

    # diameter
    diameter = nx.diameter(affiliations2_graph)
    print("Diameter:", diameter)

    # center
    center = nx.center(affiliations2_graph)
    print("Center:", center)

    # periphery
    periphery = nx.periphery(affiliations2_graph)
    print("Periphery:", periphery)

    # counting the number of triangles each node is involved in
    number_of_triangles = sum(nx.triangles(affiliations2_graph).values()) / 3
    print("Number of triangles", number_of_triangles)

    # computing the degree of each node
    node_degrees = dict(affiliations2_graph.degree())

    # printing degrees of each node
    print("Node Degrees:")
    for node, degree in node_degrees.items():
        print(f"Node {node}: Degree = {degree}")

    # computing degree distribution
    degree_counts = nx.degree_histogram(affiliations2_graph)

    # plotting degree distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(degree_counts)), degree_counts, width=0.8, color='b')
    plt.title('Κατανομή Βαθμού στο γράφημα με τα Affiliations')
    plt.xlabel('Βαθμός')
    plt.ylabel('Ακμή')
    plt.grid(True)
    plt.show()


    # plotting degree distribution as dots
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(degree_counts)), degree_counts, color='b', s=50)  # 's' parameter adjusts the size of the dots
    plt.title('Κατανομή Βαθμού στο γράφημα με τα Affiliations')
    plt.xlabel('Βαθμός')
    plt.ylabel('Ακμή')
    plt.grid(True)
    plt.show()

    # plotting degree distribution as dots
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(degree_counts)), degree_counts, color='b', s=50)  # 's' parameter adjusts the size of the dots

    # plotting curved line connecting the dots
    x = np.arange(len(degree_counts))
    y = degree_counts
    plt.plot(x, y, '-o', color='r')  # '-o' specifies line style with dots
    plt.title('Κατανομή Βαθμού στο γράφημα με τα Affiliations')
    plt.xlabel('Βαθμός')
    plt.ylabel('Ακμή')
    plt.grid(True)
    plt.show()

    # calculating the centrality
    centrality = nx.betweenness_centrality(affiliations2_graph)

    # calculating cluster coefficient for each node
    cluster_coefficient = nx.clustering(affiliations2_graph)

    # printing centrality and cluster coefficient for each node
    for node in affiliations2_graph.nodes():
        centrality_formatted = "{:.3f}".format(centrality[node])
        cluster_coefficient_formatted = "{:.3f}".format(cluster_coefficient[node])
        print(f"Node {node}: Centrality = {centrality_formatted}, Cluster Coefficient = {cluster_coefficient_formatted}")



    # finding the communities of the affiliation graph
    communities = community.best_partition(affiliations2_graph)
    from collections import defaultdict

    # initializing a defaultdict to store the nodes for each community
    community_nodes = defaultdict(list)

    # collecting nodes for each community
    for node, community_id in communities.items():
        community_nodes[community_id].append(node)

    # printing the number of elements in each community
    for community_id, nodes in community_nodes.items():
        print(f" flag4 Community {community_id}: {len(nodes)} elements")
        print(", ".join(nodes))
        print()  # Add an empty line between communities for clarity

    for node, community_id in communities.items():
        community_nodes[community_id].append(node)


    # drawing the graph with nodes colored according to their communities
    pos = nx.spring_layout(affiliations2_graph)

    # iterating over communities, draw nodes with different colors for each community
    for community_id, nodes in community_nodes.items():
        nx.draw_networkx_nodes(affiliations2_graph, pos, nodelist=nodes, node_color=plt.cm.tab10(community_id + 1), label=f"Κοινότητα {community_id + 1}")

    nx.draw_networkx_edges(affiliations2_graph, pos, alpha=0.5)
    # adding labels to the nodes
    nx.draw_networkx_labels(affiliations2_graph, pos, font_size=6)

    # showing legend
    plt.legend()

    # showing the plot
    plt.show()


    # finding all cliques in the affiliation graph
    cliques = list(nx.find_cliques(affiliations2_graph))
    print("Cliques in the graph:")
    for i, clique in enumerate(cliques, 1):
        print(f"Clique {i}: {clique}")


    # finding all cliques in the graph
    cliques = list(nx.find_cliques(affiliations2_graph))

    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]

    # finding the index of the largest clique
    max_clique_index = clique_sizes.index(max(clique_sizes))

    # finding the number of cliques
    num_cliques = len(cliques)
    print("Number of cliques in the graph:", num_cliques)

    # finding all cliques in the graph
    cliques = list(nx.find_cliques(affiliations2_graph))

    # finding the size of each clique
    clique_sizes = [len(clique) for clique in cliques]

    # finding the size of the largest clique
    max_clique_size = max(clique_sizes)

    # finding all cliques of maximum size
    max_cliques = [clique for clique in cliques if len(clique) == max_clique_size]

    # printing the number of cliques
    num_cliques = len(cliques)
    print("Number of cliques in the graph:", num_cliques)

    # printing all cliques in the graph
    print("Cliques in the graph:")
    for i, clique in enumerate(cliques):
        print(f"Clique {i + 1}: {clique}")

    # printing the largest cliques
    print(f"Largest cliques in the graph (size {max_clique_size}):")
    for i, clique in enumerate(max_cliques):
        print(f"Max Clique {i + 1}: {clique}")

    #function to calculate if the friendship paradox holds
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


    friendship_paradox_holds, paradox_nodes_count = check_friendship_paradox(affiliations2_graph)
    print("Does the friendship paradox hold in the graph?", friendship_paradox_holds)
    print("Number of nodes experiencing the paradox:", paradox_nodes_count)

    # creating a copy of the original graph to simulate bridge removal
    affiliations3_graph = affiliations2_graph.copy()

    # initializing a list to store the connected components resulting from bridge removal
    components_after_bridge_removal = []

    # iterating over the bridges and simulate their removal
    for bridge in bridges:
        # removing the bridge from the graph copy
        affiliations3_graph.remove_edge(*bridge)

        # finding connected components after bridge removal
        connected_components = list(nx.connected_components(affiliations3_graph))

        # storing the connected components
        components_after_bridge_removal.append(connected_components)

        # restoring the removed edge for further iterations
        affiliations3_graph.add_edge(*bridge)

    # printing the nodes in each connected component after removing each bridge
    for i, bridge in enumerate(bridges):
        print(f"Nodes after removing the bridge between {bridge[0]} and {bridge[1]}:")
        for j, component in enumerate(components_after_bridge_removal[i]):
            print(f"Component {j + 1}: {component}")
        print()

f.close()
