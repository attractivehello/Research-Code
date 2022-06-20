
# %%
import networkx as nx
import graphUtils 
from construction_grasp import construction_grasp

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

grid_pos, coords = graphUtils.create_layout(GG)
graphUtils.graph_function(GG, grid_pos)

# %%
district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,local_infeasible, wewe = construction_grasp(0.05, 0.3, 0.7, GG, coords, grid_pos)

import matplotlib.pyplot as plt
districts_keys = list(district.keys())
colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

color_map = {}
for node in list(GG.nodes):
    color_map[node] = "blue"
    for k in range(len(districts_keys)):
        if node in district[districts_keys[k]]:
            color_map[node] = colorss[k]
# print(color_map)

node_district = {}
for i in districts_keys:
    for j in district[i]:
        node_district[j] = i

nx.set_node_attributes(GG, values = node_district, name = "district")

nx.set_node_attributes(GG, values = color_map, name = "color")



color_map = list(color_map.values())

nx.write_graphml_lxml(GG, "generatedgraph.graphml")


plt.figure(3,figsize=(12,12))
nx.draw(GG,node_color=color_map, pos=grid_pos,with_labels = True)
plt.show()
# %%
