
import networkx as nx
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

GG = nx.read_graphml("GeneratedGraph2.graphml")



grid_pos,coords = create_layout(GG)
color_map = nx.get_node_attributes(GG, "color")

color_map = list(color_map.values())



plt.figure(3,figsize=(12,12))
nx.draw(GG,node_color=color_map, pos=grid_pos,with_labels = True)
plt.show()
