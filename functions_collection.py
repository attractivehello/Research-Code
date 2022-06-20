#Importing Libraries

import plotly.graph_objects as go
import copy as cp
from pprint import pprint
from random import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import time
from itertools import groupby,chain
import itertools
import pandas as pd
from pandas import *
import networkx as nx
import random
from random import shuffle
from itertools import chain
import string
import json
from dataclasses import dataclass
import uuid
import bisect
from datetime import datetime
import networkx.algorithms
from networkx.readwrite import json_graph



#####################################################################################################################################################################################   


#Function for getting shortest paths
def get_furthest_nodes(G):
    sp_length = {} # dict containing shortest path distances for each pair of nodes
    diameter = None # will contain the graphs diameter (length of longest shortest path)
    furthest_node_list = [] # will contain list of tuple of nodes with shortest path equal to diameter
    
    for node in G.nodes:
        # Get the shortest path from node to all other nodes
        sp_length[node] = nx.single_source_dijkstra_path_length(G,node, weight = 'distance')
        longest_path = max(sp_length[node].values()) # get length of furthest node from node
        
        # Update diameter when necessary (on first iteration and when we find a longer one)
        if diameter == None:
            diameter = longest_path # set the first diameter
            
        # update the list of tuples of furthest nodes if we have a best diameter
        if longest_path >= diameter:
            diameter = longest_path
            
            # a list of tuples containing
            # the current node and the nodes furthest from it
            node_longest_paths = [(node,other_node)
                                      for other_node in sp_length[node].keys()
                                      if sp_length[node][other_node] == longest_path]
            if longest_path > diameter:
                # This is better than the previous diameter
                # so replace the list of tuples of diameter nodes with this nodes
                # tuple of furthest nodes
                furthest_node_list = node_longest_paths
            else: # this is equal to the current diameter
                # add this nodes tuple of furthest nodes to the current list    
                furthest_node_list = furthest_node_list + node_longest_paths
                
    # return the diameter,
        # all pairs of nodes with shortest path length equal to the diameter
        # the dict of all-node shortest paths
    return({'diameter':diameter,
            'furthest_node_list':furthest_node_list,
            'node_shortest_path_dicts':sp_length})


#####################################################################################################################################################################################            

#Define a function to find the union of two lists
def Union(lst1,lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


#####################################################################################################################################################################################   

#Function for choosing centers

def plus_plus(ds, k):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """
#     np.random.seed(random_state)
    centroids = [random.choice(ds)]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)

#####################################################################################################################################################################################   

#Construction Function

def construction_grasp(delta, rcl_parameter,llambda,graph_input):
    #Choosing centers

    
    N=27
    GRID_ORIGIN = nx.grid_2d_graph(N,N)
    labels=dict(((i,j),i + (N-1-j)*N) for i, j in GRID_ORIGIN.nodes())
    # nx.relabel_nodes(DEDE,labels,False) #False=relabel the nodes in place
    inds=labels.keys()
    vals=labels.values()
    inds=[(N-j-1,N-i-1) for i,j in inds]

    #Create the dictionary of positions for the grid
    grid_pos=dict(zip(vals,inds))

    for i in list(grid_pos):
        if i not in graph_input.nodes:
            grid_pos.pop(i)

    coords = []
    for i in inds:
        for key, value in grid_pos.items():
            if i == value:
                coords.append(i)
    
    
    locations = np.array(coords)

    centroids = plus_plus(locations, 10)

    centroids = centroids.tolist()

    centroids_tuple = []
    for i in centroids:
        centroids_tuple.append(tuple((i)))

    centers_depots = []
    for i in centroids_tuple:
        for key, value in grid_pos.items():
            if i == value:
                centers_depots.append(key)
                
#     centers_depots = [526, 406, 95, 588, 678, 713, 30, 184, 449, 699]
                
    #Initialize randomized activities
    combinations = list(itertools.combinations(centers_depots, 2))



    #Calculate the average for each activity
    adjacent = {}
    for i in graph_input.nodes():
            adjacent[i] = []
    for e in graph_input.edges():
        adjacent[e[0]].append(e)
        adjacent[e[1]].append(e)

    # #Define adjacent nodes for each node

    adjacent_nodes = {}
    nodes_new = {}
    for i in adjacent:
        adjacent_nodes[i] = []
        for e in range(len(adjacent[i])):
            adjacent_nodes[i].append(adjacent[i][e][0])
            adjacent_nodes[i].append(adjacent[i][e][1])
    for i in adjacent_nodes:
        nodes_new[i] = list(set(adjacent_nodes[i]))
    adjacent_nodes = {k:[vi for vi in v if k != vi] for k,v in nodes_new.items()}



    # random.seed(2021)
    # demand= {}
    # for v in graph_input.nodes:
    #     demand[v] = random.randint(15,369)

    # random.seed(2021)
    # workload= {}
    # for v in graph_input.nodes:
    #     workload[v] = random.randint(15,89)

    # random.seed(2021)
    # n_customers= {}
    # for v in graph_input.nodes:
    #     n_customers[v] = random.randint(4,19)

#     random.seed(2021)
#     for v in centers_depots:
#         demand[v] = 400
#         workload[v] = 100
#         n_customers[v] = 20
#         for i in adjacent_nodes[v]:
#             demand[i] = random.randint(370,400)
#             workload[i] = random.randint(90,100)
#             n_customers[i] = random.randint(15,20)

    # random.seed(2021)
    # distance= {}
    # for e in graph_input.edges:
    #     distance[e] = random.randint(6,40)


    # nx.set_node_attributes(graph_input, values = n_customers, name = "n_customers")
    # nx.set_node_attributes(graph_input, values = demand, name = "demand")
    # nx.set_node_attributes(graph_input, values = workload, name = "workload")
    # nx.set_edge_attributes(graph_input, values = distance, name = "distance")
    
    nodes = list(graph_input.nodes())
    
    shortest_paths_dict = get_furthest_nodes(graph_input)['node_shortest_path_dicts']
    graph_diameter = get_furthest_nodes(graph_input)['diameter']

    total_workload = 0 
    for v in graph_input.nodes:
        total_workload = total_workload + graph_input.nodes[v]['workload']
    average_workload = total_workload/len(centers_depots)

    total_customers = 0 
    for v in graph_input.nodes:
        total_customers = total_customers + graph_input.nodes[v]['n_customers']
    average_customers = total_customers/len(centers_depots)

    total_demand = 0 
    for v in graph_input.nodes:
        total_demand = total_demand + graph_input.nodes[v]['demand']
    average_demand = total_demand/len(centers_depots)



    selected_nodes = {}
    near_nodes = {}
    for k in centers_depots:
        selected_nodes[k] = []
        selected_nodes[k] = nx.ego_graph(graph_input,k, radius = 30, center=False, undirected=True, distance='distance')
        near_nodes[k] = list(selected_nodes[k].nodes())

#     for v in near_nodes:
#         print(list(any(v in val for val in near_nodes.values())))

    #Find the percentage of selected nodes from the graph
    num_nodes = 0
    for i in near_nodes:
        num_nodes = num_nodes+len(near_nodes[i])

    percentage_nodes = 1-(num_nodes/len(nodes))

    construction_time = time.time()

    #Create the initial districts by assigning the nodes in the neighborhood to depots
    district_customers = {}
    district_workload = {}
    district_demand = {}
    unassigned = graph_input.nodes
    neighborhood = {}
    district = {}
    rcl = {}
    i = 0
    while percentage_nodes*len(graph_input.nodes) <= len(unassigned):
        for k in centers_depots:
            district_customers[k]= 0
            district_workload[k] = 0
            district_demand[k] = 0
            neighborhood[k] = []
            neighborhood[k] = near_nodes[k]
            unassigned = unassigned-set(near_nodes[k])-set(centers_depots)
            district[k] = []
            district[k] = Union(district[k], neighborhood[k])
    #Find the total of each activity for each district
            for w in district[k]:
                district_customers[k] = district_customers[k] + graph_input.nodes[w]['n_customers']
                district_workload[k] = district_workload[k] + graph_input.nodes[w]['workload']
                district_demand[k] = district_demand[k] + graph_input.nodes[w]['demand']
                


    local_infeasible = 0

    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(district_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-district_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(district_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-district_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(district_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-district_workload[centers_depots[i]],0))

    #Select a larger neighborhood for the depots
    larger_selected_nodes = {}
    larger_selected_nodes = {}
    for k in centers_depots:
        larger_selected_nodes[k] = []
        larger_selected_nodes[k] = nx.ego_graph(graph_input,k, radius = 100, center=False, undirected=True, distance='distance')
        larger_selected_nodes[k] = list(set(larger_selected_nodes[k].nodes())-set(district[k]))

    #Ensure that there is no overlap between the neighborhoods
    new_neighborhood = {}
    for k in centers_depots:
        new_neighborhood[k] = []
        for v in larger_selected_nodes[k]:
            x = list(any(v in val for val in new_neighborhood.values()))
            y = list(any(v in val for val in district.values()))
            if True not in x:
                if True not in y:             
                    new_neighborhood[k].append(v)



    #Find the infeasibility of each district
    infeasible = {}

    for k in district:
        infeasible[k] = {}
        for v in new_neighborhood[k]:
            infeasible[k][v] = (1/average_workload)*max(district_workload[k]+graph_input.nodes[v]['workload']-(1+delta)*average_workload,0)+\
                (1/average_customers)*max(district_customers[k]+graph_input.nodes[v]['n_customers']-(1+delta)*average_customers,0)+\
                    (1/average_demand)*max(district_demand[k]+graph_input.nodes[v]['demand']-(1+delta)*average_demand,0)

    obj_dispersion = max(shortest_paths_dict[x][y] for i in district for x in district[i] for y in district[i])
    frac_diameter = (1/graph_diameter)
    #Find the average dispersion of each district
    dispersion = {}
    for k in district:
        dispersion[k] = {}
        for v in new_neighborhood[k]:
            dispersion[k][v] = frac_diameter*max(obj_dispersion, max(shortest_paths_dict[x][y] for x in Union(district[k],[v]) for y in Union(district[k],[v])))

    phi = {}

    for k in district:
        phi[k] = {}
        for v in new_neighborhood[k]:
            phi[k][v] = llambda*dispersion[k][v]+(1-llambda)*infeasible[k][v]



    phi_min = {}
    for k in district:
        phi_min[k] = min(phi[k].values())

    phi_max = {}
    for k in district:
        phi_max[k] = max(phi[k].values())



    open_district = {}
    for k in district:
        open_district[k] = True

    #Create the restricted candidate list

    rcl = {}

    for k in district:
        rcl[k] = []
        if open_district[k] == True:
            for h in new_neighborhood[k]:
                if phi[k][h] <= phi_min[k]+rcl_parameter*(phi_max[k]-phi_min[k]):
                    rcl[k].append(h)

    x = 0
    r = 0
    i=0
    viable = False
    OR_OPEN = True
    RCL_EMPTY = True
    NOT_OPEN = False
    UNASSIGNED_REPEAT = False
    final_depot = False
    unassigned_length = len(unassigned)
    unassigned_previous = 0
    while ((len(unassigned) >0) and not NOT_OPEN and not UNASSIGNED_REPEAT):
        if unassigned_length == unassigned_previous:
            UNASSIGNED_REPEAT = True
#         print(unassigned_length)
        unassigned_previous = len(unassigned)
        for k in centers_depots:
            # print("First chosen depot k is")
            # print(k)
            # print("Length of RCL is")
            # print(len(rcl[k]))
            # print("The district is ")
            # print(open_district[k])

            if (len(rcl[k]) == 0):
                #print("RCL EMPTY: Going to next iteration.")
                continue

            if open_district[k]:
                for deleted in rcl[k]:
                    for i in district[k]:
                        if deleted in adjacent_nodes[i]:                        
                            if deleted in rcl[k]:
                                # print("Chosen RCL element is")
                                # print(deleted)
                                rcl[k].remove(deleted)            
                                district[k].append(deleted)
                                district_customers[k] = district_customers[k] + graph_input.nodes[deleted]['n_customers']
                                district_demand[k] = district_demand[k] + graph_input.nodes[deleted]['demand'] 
                                district_workload[k] = district_workload[k] + graph_input.nodes[deleted]['workload'] 
                                #unassigned_previous = len(unassigned)
                                if deleted in unassigned:
                                    unassigned.remove(deleted)
                                    unassigned_length = len(unassigned)
                                if (len(new_neighborhood[k]) <= 0) or (district_customers[k] >= average_customers+delta)\
                                        or (district_demand[k] >= average_demand+delta) or (district_workload[k] >= average_workload+delta):
                                    open_district[k] = False
            else:
                #print("District closed: Going to next iteration.")
                continue


        # if unassigned_length == unassigned_previous:
        #     UNASSIGNED_REPEAT = True
        # print(unassigned_length)
        # unassigned_previous = len(unassigned)




        if True not in open_district.values():
            NOT_OPEN = True

    #         RCL_EMPTY = False
    #         while len(rcl[depots[r]])<=0 and r<=len(depots):
    #             r = r+1
    #         if r <len(depots):
    #             RCL_EMPTY = True

#     for k in centers_depots:
#         for i in district[k]:
#             print(list(any(i in val for val in district.values())))




    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                if v in adjacent_nodes[x]:
                    if open_district[k] == True:
                        unique_pls = list(any(v in val for val in district.values()))
                        if True not in unique_pls:
                            district[k].append(v)
                            unassigned.remove(v)
                            district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                            district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                            district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                            if (district_customers[k] >= average_customers+delta) or (district_demand[k] >= average_demand+delta) \
                                or (district_workload[k] >= average_workload+delta):
                                open_district[k] = False

                                

    
    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                if v in adjacent_nodes[x]:
                    unique_pls = list(any(v in val for val in district.values()))
                    if True not in unique_pls:
                        district[k].append(v)
                        unassigned.remove(v)
                        district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                        district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                        district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                        if (district_customers[k] >= average_customers+delta) or (district_demand[k] >= average_demand+delta) \
                            or (district_workload[k] >= average_workload+delta):
                            open_district[k] = False

                            
    unassigned = list(unassigned)
    
    while len(unassigned)>0:
        for k in centers_depots:
            for x in district[k]:
                for v in unassigned:
                    if v in adjacent_nodes[x]:
                        unique_pls = list(any(v in val for val in district.values()))
                        if True not in unique_pls:
                            district[k].append(v)
                            unassigned.remove(v)
                            district_customers[k] = district_customers[k] + graph_input.nodes[v]['n_customers']
                            district_demand[k] = district_demand[k] + graph_input.nodes[v]['demand'] 
                            district_workload[k] = district_workload[k] + graph_input.nodes[v]['workload'] 
                                
    districts_keys = list(district.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in district[districts_keys[k]]:
                color_map[node] = colorss[k]
    

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()

    local_infeasible = 0

    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(district_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-district_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(district_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-district_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(district_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-district_workload[centers_depots[i]],0))

    #print(local_infeasible)
    construction_obj = max(shortest_paths_dict[x][y] for i in district for x in district[i] for y in district[i])
    
    return district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,local_infeasible

#####################################################################################################################################################################################   


#LS-NBI 

def localsearch_node_best_improvement(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
        
        
        
    def update_merit_function(input_solution,k,l,bu):
        x= cp.deepcopy(input_solution)
        x[l].remove(bu)
        x[k].append(bu)

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

        #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        solution_infeasibility = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            solution_infeasibility = solution_infeasibility + (ga1_best+ga2_best+ga3_best)
        

        solution_objective = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))


        weight_district_best = llambda*solution_objective + (1-llambda)*solution_infeasibility

        return weight_district_best
    

      
    node_district_matching = {}
    for k in district:
        for i in district[k]:
            node_district_matching[i] = k


     
    moves = {}
    for depots in centers_depots:
        moves[depots] = []
        for nodes in node_district_matching:
            if node_district_matching[nodes] != depots:
                moves[depots].append(nodes)
#     print(moves)

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
            
    current_best_objective = merit_function(district)

    current_best_solution = cp.deepcopy(district)

    while(nmoves<1000 and local_optima==False):
        
#         print("Repeating the for loop again")
#         print("The current best objective is",current_best_objective)
        improvement = False
        for k in centers_depots:
#             print("Moving to next center",k)
            improvement = False
            for basic_unit in moves[k]:
                if nx.has_path(graph_input.subgraph(current_best_solution[k]+[k,basic_unit]),k,basic_unit) == True:
 
                    new_objective = update_merit_function(current_best_solution,k,node_district_matching[basic_unit],basic_unit)

                    if new_objective < current_best_objective:
    #                     print(k)
#                         print("Better Objective found")
                        best_basic_unit = basic_unit
#                         print(best_basic_unit)
                        current_best_objective = new_objective
                        improvement = True
            
            if improvement == True:
                current_best_solution[k].append(best_basic_unit)
                current_best_solution[node_district_matching[best_basic_unit]].remove(best_basic_unit)
#                 print("Succeeded in improving", current_best_objective)
#                 print("The best basic unit is", best_basic_unit)
                node_district_matching[best_basic_unit] = k
                moves[k].remove(best_basic_unit)
                moves[node_district_matching[best_basic_unit]].append(best_basic_unit)
                   


#             moves = {}
#             for depotss in centers_depots:
#                 moves[depotss] = []
#                 for nodes in node_district_matching:
#                     if node_district_matching[nodes] != depotss:
#                         moves[depotss].append(nodes)

        
        
        if improvement == True:    
            nmoves = nmoves+1
            local_optima = False
        else:
            local_optima = True
            print("Local Optimum Reached")
            
            
            
    best_obj = max(shortest_paths_dict[a][b] for i in current_best_solution for a in current_best_solution[i]+[i] for b in current_best_solution[i]+[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in current_best_solution[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, current_best_solution

#####################################################################################################################################################################################
#LS-NFI

def localsearch_node_first_improvement(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
        
        
        
    def update_merit_function(input_solution,k,l,bu):
        x= cp.deepcopy(input_solution)
        x[l].remove(bu)
        x[k].append(bu)

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

        #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        solution_infeasibility = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            solution_infeasibility = solution_infeasibility + (ga1_best+ga2_best+ga3_best)
        

        solution_objective = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))


        weight_district_best = llambda*solution_objective + (1-llambda)*solution_infeasibility

        return weight_district_best
    
    
    district_trial2 = {}
    best_sol = {}

    
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]
        
    node_district_matching = {}
    for k in district:
        for i in district[k]:
            node_district_matching[i] = k


     
    moves = {}
    for depots in centers_depots:
        moves[depots] = []
        for nodes in node_district_matching:
            if node_district_matching[nodes] != depots:
                moves[depots].append(nodes)
#     print(moves)

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
            
    current_best_objective = merit_function(district)

    current_best_solution = cp.deepcopy(district)

    while(nmoves<1000 and local_optima==False):
        
#         print("Repeating the for loop again")
#         print(current_best_objective)
        improvement = False
        for k in centers_depots:
#             print("Moving to next center",k)
            improvement = False
            basic_unit = 0
#             print("The basic unit is", moves[k][basic_unit])
            while improvement == False and basic_unit <= len(moves[k])-1:
#                 print(moves[k])
#                 print("The basic unit is", moves[k][basic_unit])
#                 for stuff in current_best_solution[k]+[k]:
#                     if moves[k][basic_unit] in adjacent_nodes[stuff]:
                if nx.has_path(graph_input.subgraph(current_best_solution[k]+[k,moves[k][basic_unit]]),k,moves[k][basic_unit]) == True:

                                          
                    new_objective = update_merit_function(current_best_solution,k,node_district_matching[moves[k][basic_unit]],moves[k][basic_unit])
                    if new_objective < current_best_objective:
    #                     print(k)
#                         print("Better Objective found")
                        best_basic_unit = moves[k][basic_unit]
#                         print(best_basic_unit)
                        current_best_objective = new_objective
                        improvement = True
                            
                basic_unit = basic_unit +1
#                 print("Basic unit number", basic_unit)

                if improvement == True:
                    current_best_solution[k].append(best_basic_unit)
                    current_best_solution[node_district_matching[best_basic_unit]].remove(best_basic_unit)
#                     print("Succeeded in improving", current_best_objective)
#                     print("The best basic unit is", best_basic_unit)
                    node_district_matching[best_basic_unit] = k
                    moves[k].remove(best_basic_unit)
                    moves[node_district_matching[best_basic_unit]].append(best_basic_unit)
                    #Three centers equal distribution


        if improvement == True:    
            nmoves = nmoves+1
            local_optima = False
        else:
            local_optima = True
            print("Local Optimum Reached")
            
            
            
    best_obj = max(shortest_paths_dict[a][b] for i in current_best_solution for a in current_best_solution[i]+[i] for b in current_best_solution[i]+[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in current_best_solution[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, current_best_solution


#####################################################################################################################################################################################
#LS-DBI

def localsearch_depot_best_improvement(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
        
        
        
    def update_merit_function(input_solution,k,l,bu):
        x= cp.deepcopy(input_solution)
        x[l].remove(bu)
        x[k].append(bu)

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

        #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        solution_infeasibility = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            solution_infeasibility = solution_infeasibility + (ga1_best+ga2_best+ga3_best)
        

        solution_objective = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))


        weight_district_best = llambda*solution_objective + (1-llambda)*solution_infeasibility

        return weight_district_best
    

 
        
    node_district_matching = {}
    for k in district:
        for i in district[k]:
            node_district_matching[i] = k
#     print(node_district_matching[444])

     
    moves = {}
    for depots in centers_depots:
        moves[depots] = []
        for nodes in node_district_matching:
            if node_district_matching[nodes] != depots:
                moves[depots].append(nodes)
#     print(moves)

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
            
    current_best_objective = merit_function(district)

    current_best_solution = cp.deepcopy(district)
    
    
    
    while(nmoves<1000 and local_optima == False):
    
#         print("Repeating for loop again")
#         print("The current best objective is", current_best_objective)
        improvement = False
        for basic_unit in list(set(graph_input.nodes)-set(centers_depots)):
#             improvement = False
            center_removed = node_district_matching[basic_unit]
#             print("Basic unit is", basic_unit)
#             print("back here again")
            center_improvement = False
            for centers_to in centers_depots:
                if center_removed != centers_to: 
                    new_objective  = update_merit_function(current_best_solution,centers_to,center_removed,basic_unit)

                    if new_objective < current_best_objective:
#                         print("Better Objective Found", new_objective)
                        best_center = centers_to
                        current_best_objective = new_objective
                        center_improvement = True
#             print(center_improvement)


            if center_improvement == True:
#                 if nx.has_path(graph_input.subgraph(current_best_solution[best_center]+[best_center,basic_unit]),best_center,basic_unit) == True:
                current_best_solution[best_center].append(basic_unit)
                current_best_solution[center_removed].remove(basic_unit)
#                 print("Succeeded in improving", current_best_objective)
#                 print("The best center is", best_center)
                node_district_matching[basic_unit] = best_center
                improvement = True
        
#         print("We testing stuff here")
#         print("Improvement is", improvement)
        if improvement == True:
            nmoves = nmoves+1
            local_optima = False
        else:
            local_optima = True
            print("Local Optimum Reached")
            
            
            
    best_obj = max(shortest_paths_dict[a][b] for i in current_best_solution for a in current_best_solution[i]+[i] for b in current_best_solution[i]+[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in current_best_solution[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, current_best_solution


#####################################################################################################################################################################################
#LS-DFI

def localsearch_depot_first_improvement(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
   
    #Save value of best solution for comparisons. Onlly calculate the new one. Update when you find a better solution.
    #Update membership of each district. 
    
    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    
    def merit_function(x):

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))

        weight_district_best_g = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)

        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return value of merit function

        return weight_district_best
        
        
        
        
    def update_merit_function(input_solution,k,l,bu):
        x= cp.deepcopy(input_solution)
        x[l].remove(bu)
        x[k].append(bu)

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

        #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']

        solution_infeasibility = 0
        for i in range(len(centers_depots)):

            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            solution_infeasibility = solution_infeasibility + (ga1_best+ga2_best+ga3_best)
        

        solution_objective = (max(shortest_paths_dict[a][b] for i in x for a in x[i]+[i] for b in x[i]+[i]))


        weight_district_best = llambda*solution_objective + (1-llambda)*solution_infeasibility

        return weight_district_best
    
    
    district_trial2 = {}
    best_sol = {}

    
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]
        
    node_district_matching = {}
    for k in district:
        for i in district[k]:
            node_district_matching[i] = k


     
    moves = {}
    for depots in centers_depots:
        moves[depots] = []
        for nodes in node_district_matching:
            if node_district_matching[nodes] != depots:
                moves[depots].append(nodes)
#     print(moves)

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    move_index = 0
    
            
    current_best_objective = merit_function(district)

    current_best_solution = cp.deepcopy(district)
    
    
    
    while(nmoves<1000 and local_optima == False):
    
#         print("Repeating for loop again")
#         print("The current best objective is", current_best_objective)
        improvement = False
        for basic_unit in list(set(graph_input.nodes)-set(centers_depots)):
            improvement = False
            center_removed = node_district_matching[basic_unit]
            center_visited = 0
            while improvement == False and center_visited <= len(centers_depots)-2:
                centers_to = centers_depots[center_visited]
                new_objective  = update_merit_function(current_best_solution,centers_to,center_removed,basic_unit)

                if new_objective < current_best_objective:
#                     print("Better Objective Found")
                    best_center = centers_to
                    current_best_objective = new_objective
                    improvement = True

                
                if improvement == True:
                    current_best_solution[best_center].append(basic_unit)
                    current_best_solution[center_removed].remove(basic_unit)
#                     print("Succeeded in improving", current_best_objective)
#                     print("The best center is", best_center)
                    node_district_matching[basic_unit] = best_center
  
                
                center_visited = center_visited +  1
#                 print(center_visited)
                centers_to = centers_depots[center_visited]

                
        if improvement == True:
            nmoves = nmoves+1
            local_optima = False
        else:
            local_optima = True
            print("Local Optimum Reached")
            
            
            
    best_obj = max(shortest_paths_dict[a][b] for i in current_best_solution for a in current_best_solution[i]+[i] for b in current_best_solution[i]+[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in current_best_solution[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, current_best_solution


#####################################################################################################################################################################################
#GRASP-LS

def localsearch_grasp(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input):    
    def decision(x,y):

        y_customers = {}
        y_demand = {}
        y_workload = {}

        x_customers = {}
        x_demand = {}
        x_workload = {}

        for k in centers_depots:
            y_customers[k] = 0
            y_demand[k] = 0
            y_workload[k] = 0
            x_customers[k] = 0
            x_demand[k] = 0
            x_workload[k] = 0

    #Calculate the total for each  activity measure for each district allocation
        for k in centers_depots:
            for w in y[k]:
                y_customers[k] = y_customers[k] + graph_input.nodes[w]['n_customers']
                y_workload[k] = y_workload[k] + graph_input.nodes[w]['workload']
                y_demand[k] = y_demand[k] + graph_input.nodes[w]['demand']
            for w in x[k]:
                x_customers[k] = x_customers[k] + graph_input.nodes[w]['n_customers']
                x_workload[k] = x_workload[k] + graph_input.nodes[w]['workload']
                x_demand[k] = x_demand[k] + graph_input.nodes[w]['demand']






        weight_district_temp_f = (max(shortest_paths_dict[a][b] for i in y for a in y[i] for b in y[i]))
        #weight_district_temp_f = llambda*(max(get_furthest_nodes(graph_input.subgraph(y[centers_depots[i]]))['diameter'] for i in range(len(centers_depots))))

        weight_district_temp_g = 0
        for i in range(len(centers_depots)):

            ga1_temp = ((1/average_customers)*max(y_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-y_customers[centers_depots[i]],0))
            ga2_temp = ((1/average_demand)*max(y_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-y_demand[centers_depots[i]],0))
            ga3_temp = ((1/average_workload)*max(y_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-y_workload[centers_depots[i]],0))

            weight_district_temp_g = weight_district_temp_g +(ga1_temp+ga2_temp+ga3_temp)

        weight_district_temp = llambda*weight_district_temp_f + (1-llambda)*weight_district_temp_g





        weight_district_best_f = (max(shortest_paths_dict[a][b] for i in x for a in x[i] for b in x[i]))
        weight_district_best_g = 0
        for i in range(len(centers_depots)):
            
            ga1_best = ((1/average_customers)*max(x_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-x_customers[centers_depots[i]],0))
            ga2_best = ((1/average_demand)*max(x_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-x_demand[centers_depots[i]],0))
            ga3_best = ((1/average_workload)*max(x_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-x_workload[centers_depots[i]],0))

            weight_district_best_g = weight_district_best_g + (ga1_best+ga2_best+ga3_best)



        weight_district_best = llambda*weight_district_best_f + (1-llambda)*weight_district_best_g

    #Return True if the first district allocation is better

        if(weight_district_temp<weight_district_best):

            return True
        else:

            return False

    district_trial2 = {}
    best_sol = {}

#     print(district_trial2)
#     print(best_sol)
    #Copy the current district allocation
    for i in range(len(centers_depots)):
        district_trial2[centers_depots[i]] = district[centers_depots[i]][:]

    for i in range(len(centers_depots)):
        best_sol[centers_depots[i]] = district[centers_depots[i]][:]

    #Initialize a dictionary of moves
    moves = {}

    for i in range(len(centers_depots)):

        moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

    for i in moves:
        moves[i] = list(itertools.chain(*moves[i]))

    depo_choices = {}
    for k in centers_depots:
        depo_choices[k] = [item for item in combinations
                if item[0] == k or item[1] == k]


    choose = random.choice(range(len(centers_depots)))

    k = centers_depots[choose]
    chosen_depots = random.choice(depo_choices[k])
    kend = None
    nmoves = 0
    p = len(centers_depots)
    local_optima = False

    number_of_moves = 0
    
    while(nmoves<1000 and local_optima==False):
#         print(".....")
#         print(local_optima)
#         print(".....")
        improvement = False
        while((len(moves[k])>0) and (improvement == False)):
            move_to = random.choice(moves[k])
            moves[k].remove(move_to)
            number_of_moves = number_of_moves + 1
#             print("Length of moves is")
#             print(number_of_moves)

#             print(k)
#             print(len(moves[k]))
            for i in district_trial2[k]:
                if move_to in adjacent_nodes[i]:
                    for f in centers_depots:
                        if move_to in district_trial2[f]:
                            district_trial2[f].remove(move_to)

                        unique_districts = list(any(move_to in val for val in district_trial2.values()))
                        if True not in unique_districts:
                            if move_to in adjacent_nodes[i]:
                                district_trial2[k].append(move_to)

#             paths_list = []
#             for nodes in district[k]:
#                 paths_list.append(nx.has_path(graph_input.subgraph(district[k]+[k]),k,nodes))
# #             if False in paths_list:
# #                 for cent in range(len(centers_depots)):
# #                     district_trial2[centers_depots[cent]] = best_sol[centers_depots[cent]][:]
#             print("guess not")
#             if False not in paths_list:
#                 moves[k].remove(move_to)
#     #Use the decision() function to check whether the performed move is an improvement              
            if(decision(best_sol,district_trial2)==True):
#                 print("Yes")
                best_sol = {}
                for i in range(len(centers_depots)):
                    best_sol[centers_depots[i]] = district_trial2[centers_depots[i]][:]
                nmoves = nmoves+1
#                 print(nmoves)
                #print(decision(best_sol,district_trial2))
                improvement = True
                choose = (choose+1) % p
                kend = k
                k = centers_depots[choose]
                chosen_depots = random.choice(depo_choices[k])
                
                
#                 for i in range(len(centers_depots)):

#                     moves[centers_depots[i]]= [v for k,v in district.items() if k != centers_depots[i]]

#                 for i in moves:
#                     moves[i] = list(itertools.chain(*moves[i]))
#                     for i in range(len(centers_depots)):
#                         if move_to in moves[centers_depots[i]]:
#                             moves[centers_depots[i]].remove(move_to)
            else:
#                 print("No")
                #print(decision(best_sol,district_trial2))
                for i in range(len(centers_depots)):
                    district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]
                #small chance of sad infinite loop :(
        if improvement == False:
            choose = (choose+1) % p
            k = centers_depots[choose]
            chosen_depots = random.choice(depo_choices[k])
#         #         if len(moves[k]) == 0:
#         #             choose = (choose+1) % p
#         #             k = centers_depots[choose]
#         #             chosen_depots = random.choice(depo_choices[k])

        if k == kend:
            local_optima = True
            print("Local Optimum Reached")
#     #     for k in centers_depots:
    #         for i in best_sol[k]:
    #             print(list(any(i in val for val in best_sol.values())))          
#             else:
#                 for i in range(len(centers_depots)):
#                     district_trial2[centers_depots[i]] = best_sol[centers_depots[i]][:]

    districts_keys = list(best_sol.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in best_sol[districts_keys[k]]:
                color_map[node] = colorss[k]

    color_map = list(color_map.values())

#     plt.figure(3,figsize=(12,12))
#     nx.draw(graph_input,node_color=color_map, pos=grid_pos,with_labels = True)
#     plt.show()
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


#####################################################################################################################################################################################
#Shaking

def shaking(district,centers_depots,combinations,adjacent_nodes,average_customers, average_demand,average_workload,shortest_paths_dict,llambda,delta,graph_input,shaking_steps):    
    
    shaking_input = cp.deepcopy(district)

#     selections = cp.deepcopy(centers_depots)
    
    selections = []
    for i in shaking_input:
        if len(shaking_input[i])>shaking_steps:
            selections.append(i)
    depot_from = random.choice(selections)
    selections.remove(depot_from)
    depot_to = random.choice(selections)

    for step in range(shaking_steps):
        node_to_move = random.choice(shaking_input[depot_from])
        shaking_input[depot_to].append(node_to_move)
        shaking_input[depot_from].remove(node_to_move)

    best_sol = cp.deepcopy(shaking_input)
    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))

    #print(local_infeasible)
    
    return best_obj, local_infeasible, best_sol


#####################################################################################################################################################################################
#Match centers of different solutions

def center_matching(c1,c2):
    c1 = list(c1)
    c2 = list(c2)
    
    c1 = list(set(c1).difference(c2))
    c2= list(set(c2).difference(c1))
    

    B = nx.Graph()
    B.add_nodes_from(c1, bipartite = 0)
    B.add_nodes_from(c2, bipartite = 1)
    for n in c1:
        B.add_edges_from([(n,v) for v in c2])

    distances_dict = {}
    for i in list(B.edges):
        distances_dict[i] = shortest_paths_dict[i[0]][i[1]]


    
    nx.set_edge_attributes(B, distances_dict, "distance")


    yeye = dict(itertools.islice(networkx.algorithms.bipartite.matching.minimum_weight_full_matching(B, weight = "distance").items(), len(networkx.algorithms.bipartite.matching.minimum_weight_full_matching(B, weight = "distance").items()) // 2))

    
    return yeye


#####################################################################################################################################################################################
#Find distance between solutions

def dsol(matchings,solution1,solution2):
    distance_between_solutions = 0
    for i in matchings:

        for node in solution1[i]:
            if i not in solution2[matchings[i]]:

                distance_between_solutions = distance_between_solutions + 1

    distance_between_solutions = distance_between_solutions/(500-(500/int(ceil(len(matchings)))))
    
    return distance_between_solutions


#####################################################################################################################################################################################
#Initiate set of possible moves

def moves_set(solution1,solution2):
    solutions_matching = center_matching(solution1.allocation.keys(),solution2.allocation.keys())
    set_of_moves = {}
    for cent in solutions_matching:
        set_of_moves[solutions_matching[cent]] = []
        
    for i in solutions_matching:
        for node in solution1.allocation[i]:
            if i not in solution2.allocation[solutions_matching[i]]:
                set_of_moves[solutions_matching[i]].append(node)
    
    set_of_moves = dict( [(k,v) for k,v in set_of_moves.items() if len(v)>0])     
    
    return set_of_moves


#####################################################################################################################################################################################
#Match nodes to districts

def matching_nodes_districts(input_solution):
    nk_matching = {}
    for items in input_solution:
        for node in input_solution[items]:
            nk_matching[node] = items   
            
    return nk_matching



#####################################################################################################################################################################################
#Function for performing PR Moves

def pr_move_function(sol, sol_moves, node_matchings, graph_input):
    
    best_sol = cp.deepcopy(sol)
    sol_moves_copy = cp.deepcopy(sol_moves)
    node_matchings_copy = cp.deepcopy(node_matchings)
    
    adjacent = {}
    for i in graph_input.nodes():
            adjacent[i] = []
    for e in graph_input.edges():
        adjacent[e[0]].append(e)
        adjacent[e[1]].append(e)

    #Define adjacent nodes for each node

    adjacent_nodes = {}
    nodes_new = {}
    for i in adjacent:
        adjacent_nodes[i] = []
        for e in range(len(adjacent[i])):
            adjacent_nodes[i].append(adjacent[i][e][0])
            adjacent_nodes[i].append(adjacent[i][e][1])
    for i in adjacent_nodes:
        nodes_new[i] = list(set(adjacent_nodes[i]))
    adjacent_nodes = {k:[vi for vi in v if k != vi] for k,v in nodes_new.items()}
    
#     for st in best_sol:
#         print(len(best_sol[st]))
    
#     print(best_sol.keys())

    for k in sol_moves_copy:
        for move_to in sol_moves_copy[k]:
            adjacency = False
            for nodes in best_sol[k]:
                if move_to in adjacent_nodes[nodes]:
                    adjacency= True
                if adjacency == True:
                    if move_to not in list(best_sol.keys()):
                        if move_to in node_matchings_copy:
        #                     print("Move was made")
        #                     print(move_to)
        #                     print(node_matchings_copy[move_to])
                #                     print("The move was in ", node_matchings_copy[move_to])
                            best_sol[k].append(move_to)
                            best_sol[node_matchings_copy[move_to]].remove(move_to)
                            node_matchings_copy[move_to] = k

    
    best_obj = max(shortest_paths_dict[a][b] for i in best_sol for a in best_sol[i] for b in best_sol[i])
    
    best_sol_customers = {}
    best_sol_workload = {}
    best_sol_demand = {}
    
    centers_depots = list(best_sol.keys())
    for k in centers_depots:
        best_sol_customers[k]=0
        best_sol_demand[k]=0
        best_sol_workload[k]=0
        
    for k in centers_depots:
        for w in best_sol[k]:
            best_sol_customers[k] = best_sol_customers[k] + graph_input.nodes[w]['n_customers']
            best_sol_workload[k] = best_sol_workload[k] + graph_input.nodes[w]['workload']
            best_sol_demand[k] = best_sol_demand[k] + graph_input.nodes[w]['demand']
    
    local_infeasible = 0
    
    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(best_sol_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-best_sol_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(best_sol_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-best_sol_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(best_sol_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-best_sol_workload[centers_depots[i]],0))
        
    return best_obj, local_infeasible, best_sol



#####################################################################################################################################################################################
#The actual PR Algorithm

def PR_ALGORITHM(inp_sol):
    best_solution = cp.deepcopy(inp_sol[0].allocation)
    merit_best = llambda *inp_sol[0].obj + (1-llambda)*inp_sol[0].inf
    objective_best = inp_sol[0].obj
    inf_best = inp_sol[0].inf
    
    for i in range(0,14):
        for j in range(i+1,15):
            node_district_matching_i = matching_nodes_districts(inp_sol[i].allocation)
#         print(pr_input_solutions[i].allocation)
            node_district_matching_j = matching_nodes_districts(inp_sol[j].allocation)

            moves_solution_i = moves_set(inp_sol[j],inp_sol[i])
            moves_solution_j = moves_set(inp_sol[i],inp_sol[j])

            solution_i = cp.deepcopy(inp_sol[i].allocation)
            solution_j = cp.deepcopy(inp_sol[j].allocation)

            objective_i, inf_i, sol_i = pr_move_function(inp_sol[i].allocation, moves_solution_i, node_district_matching_i, G3)
            objective_j, inf_j, sol_j = pr_move_function(inp_sol[j].allocation, moves_solution_j, node_district_matching_j, G3)

            if objective_i < objective_j:
                intermediate_objective = objective_i
                intermediate_inf = inf_i
                intermediate_sol = cp.deepcopy(sol_i)
            else:
                intermediate_objective = objective_j
                intermediate_inf = inf_j
                intermediate_sol = cp.deepcopy(sol_j)

            merit_intermediate = llambda*intermediate_objective + (1-llambda)*intermediate_inf

#             print(merit_intermediate)

            if merit_intermediate < merit_best:
                best_solution = cp.deepcopy(intermediate_sol)
                merit_best = merit_intermediate
                objective_best = intermediate_objective
                inf_best = intermediate_inf
                
    return objective_best, inf_best, best_solution

#####################################################################################################################################################################################