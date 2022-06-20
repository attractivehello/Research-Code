import networkx as nx
import random
import matplotlib.pyplot as plt

def create_layout(graph, **kwargs):
    """
    Creates a layout for the graph using the graphviz layout engine.
    """
    N=27
    GRID_ORIGIN = nx.grid_2d_graph(N,N)
    labels=dict(((i,j),i + (N-1-j)*N) for i, j in GRID_ORIGIN.nodes())
    # nx.relabel_nodes(DEDE,labels,False) #False=relabel the nodes in place
    inds=labels.keys()
    vals=labels.values()
    inds=[(N-j-1,N-i-1) for i,j in inds]

    #Create the dictionary of positions for the grid
    grid_pos=dict(zip(vals,inds))

    #turn the keys in grid_pos to strings
    grid_pos = {str(k):v for k,v in grid_pos.items()}

    for i in list(grid_pos):
        if i not in graph.nodes:
            grid_pos.pop(i)
    coords = []
    for i in inds:
        for key, value in grid_pos.items():
            if i == value:
                coords.append(i)

    return grid_pos,coords

def graph_function(graph_input,positions):
    grid_pos = create_layout(graph_input)
    plt.figure(3,figsize=(12,12))
    nx.draw(graph_input, pos = positions, with_labels = True)
    plt.show()

def graph_attributes_selection(graph_input,seeds):
    random.seed(seeds)
    demand= {}
    for v in graph_input.nodes:
        demand[v] = random.randint(15,400)

    random.seed(seeds)
    workload= {}
    for v in graph_input.nodes:
        workload[v] = random.randint(15,100)

    random.seed(seeds)
    n_customers= {}
    for v in graph_input.nodes:
        n_customers[v] = random.randint(4,20)


    random.seed(seeds)
    distance= {}
    for e in graph_input.edges:
        distance[e] = random.randint(5,20)


    nx.set_node_attributes(graph_input, values = n_customers, name = "n_customers")
    nx.set_node_attributes(graph_input, values = demand, name = "demand")
    nx.set_node_attributes(graph_input, values = workload, name = "workload")
    nx.set_edge_attributes(graph_input, values = distance, name = "distance")

def update_graph_attributes(graph_input,demand_input,workload_input,n_customers_input,distance_input):

    varied_customers = {key:random.randint(int((3*n_customers_input[key])/4),int((5*n_customers_input[key])/4)) for key in n_customers_input}
    varied_workload = {key:random.randint(int((3*workload_input[key])/4),int((5*workload_input[key])/4)) for key in workload_input}
    varied_demand = {key:random.randint(int((3*demand_input[key])/4),int((5*demand_input[key])/4)) for key in demand_input}

    nx.set_node_attributes(graph_input, values = varied_customers, name = "n_customers")
    nx.set_node_attributes(graph_input, values = varied_demand, name = "demand")
    nx.set_node_attributes(graph_input, values = varied_workload, name = "workload")
    nx.set_edge_attributes(graph_input, values = distance_input, name = "distance")


