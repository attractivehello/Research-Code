
# %%
import networkx as nx
import graphUtils 

GG = nx.read_graphml("basegraphattributes.graphml")

# %%

base_customers = nx.get_node_attributes(GG, "n_customers")
base_workload = nx.get_node_attributes(GG, "workload")
base_demand = nx.get_node_attributes(GG, "demand")
base_distance = nx.get_edge_attributes(GG, "distance")

graphUtils.update_graph_attributes(GG, base_demand, base_workload, base_customers, base_distance)

# %%
nx.write_graphml_lxml(GG, "generatedgraph.graphml")
# %%

grid_pos = graphUtils.create_layout(GG)
graphUtils.graph_function(GG, grid_pos)

# %%
