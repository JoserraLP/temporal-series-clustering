from pyvis.network import Network

from temporal_series_clustering.sheaf.sheaf import Sheaf, Vertex

import string
import itertools
import networkx as nx


# Method for combinations of data sources
def identity(a):
    """
    Identity function

    :param a: input value
    :return: input value
    """
    return a


def create_sheaf_model(input_vertices_labels: list):
    """
    Create the sheaf model with the vertices.

    The model will have 2-higher level faces, the first one interconnecting all the vertices by pairs and
    the second one interconnecting all face-one level vertices on a single output vertex noted '@'.

    All the functions are identity.

    :param input_vertices_labels: input vertices labels
    :type input_vertices_labels: list
    :return: sheaf model with the specified architecture
    """
    # Generate all possible relations (combinations of 2) between the strings
    face_one_relations = list(itertools.combinations(input_vertices_labels, 2))

    # Set vertices literals
    vertices = {item: Vertex(item) for item in input_vertices_labels}

    # Build sheaf base space
    sheaf = Sheaf()

    # Build 0-faces/vertices
    sheaf.set_vertices(list(vertices.values()))

    # Build 1-faces
    face_one_faces = {}
    for relation in face_one_relations:
        item1, item2 = relation
        face_one_faces[item1 + '_' + item2] = {
            "subfaces": {item1: vertices[item1], item2: vertices[item2]},
            "restriction_map": {item1: identity, item2: identity}
        }

    sheaf.set_higher_faces(face_one_faces)

    # Build 2-faces
    # First retrieve the variables
    face_two_vertices = {item: sheaf.get_face(item) for item in list(face_one_faces.keys())}

    face_two_faces = {'@': {
        "subfaces": {k: v for k, v in face_two_vertices.items()},
        "restriction_map": {k: identity for k in list(face_two_vertices.keys())}
    }}

    sheaf.set_higher_faces(face_two_faces)
    return sheaf


def create_simplified_graph(num_vertices: int, instant_consistency: dict) -> nx.Graph:
    """
    Create a simplified version of the sheaf model where the connections are only for base vertices.
    
    :param num_vertices: number of vertices from the full graph
    :param instant_consistency: instant value for consistencies

    :return:
    """
    # List of strings representing alphabet on uppercase
    input_vertices_labels = list(string.ascii_uppercase)[:num_vertices]

    # Generate all possible relations (combinations of 2) between the strings
    combinations = list(itertools.combinations(input_vertices_labels, 2))

    # Create the graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(input_vertices_labels)

    # Add edges to the graph with the instant value consistency
    for combination in combinations:
        # Round to six, but it should be parametrized
        G.add_edge(combination[0], combination[1], consistency=round(instant_consistency['_'.join(combination)], 6))
    return G


def show_network(sheaf):
    """
    Show a sheaf model

    :param sheaf: sheaf model
    :return:
    """
    nt = Network()
    # populates the nodes and edges data structures
    nt.from_nx(sheaf.visualize_sheaf_as_digraph())
    nt.show('nx.html', notebook=False)
