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
import functions_collection as fnc

def construction_grasp(delta, rcl_parameter,llambda,graph_input, coords, grid_pos):
    #Choosing centers
   
    locations = np.array(coords)
    print(locations)

    centroids = fnc.plus_plus(locations, 10)

    centroids = centroids.tolist()

    centroids_tuple = []
    for i in centroids:
        centroids_tuple.append(tuple((i)))

    centers_depots = []
    for i in centroids_tuple:
        for key, value in grid_pos.items():
            if i == value:
                centers_depots.append(key)
                
    centers_depots = ["30","103","217","543","633","667","401","365","92","277"]

    
    
                
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


    nodes = list(graph_input.nodes())
    
    shortest_paths_dict = fnc.get_furthest_nodes(graph_input)['node_shortest_path_dicts']
    graph_diameter = fnc.get_furthest_nodes(graph_input)['diameter']

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
            district[k] = fnc.Union(district[k], neighborhood[k])
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
            dispersion[k][v] = frac_diameter*max(obj_dispersion, max(shortest_paths_dict[x][y] for x in fnc.Union(district[k],[v]) for y in fnc.Union(district[k],[v])))

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
        unassigned_previous = len(unassigned)
        for k in centers_depots:

            if (len(rcl[k]) == 0):
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
                continue






        if True not in open_district.values():
            NOT_OPEN = True




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
                            
    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                
                
                if v in adjacent_nodes[k]:
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
                            
    a =  0
    unassigned = list(unassigned)
    for k in centers_depots:
        for x in district[k]:
            for v in unassigned:
                
                
                if v in adjacent_nodes[k]:
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

                            

                                
    districts_keys = list(district.keys())
    colorss = ["lightcoral","sandybrown","darkorange","lawngreen","green","aqua","steelblue","violet","purple","maroon"]

    color_map = {}
    for node in list(graph_input.nodes):
        color_map[node] = "blue"
        for k in range(len(districts_keys)):
            if node in district[districts_keys[k]]:
                color_map[node] = colorss[k]
    

    color_map = list(color_map.values())

    local_infeasible = 0

    for i in range(len(centers_depots)):
        local_infeasible = local_infeasible + ((1/average_customers)*max(district_customers[centers_depots[i]]-(1+delta)*average_customers,(1-delta)*average_customers-district_customers[centers_depots[i]],0))+\
            ((1/average_demand)*max(district_demand[centers_depots[i]]-(1+delta)*average_demand,(1-delta)*average_demand-district_demand[centers_depots[i]],0))+\
                ((1/average_workload)*max(district_workload[centers_depots[i]]-(1+delta)*average_workload,(1-delta)*average_workload-district_workload[centers_depots[i]],0))

    #append district key to district
    for i in district:
        district[i].insert(0,i)

    #rename keys from 0 to 10
    district = dict(zip(range(len(district)),district.values()))

    construction_obj = max(shortest_paths_dict[x][y] for i in district for x in district[i] for y in district[i])
    
    return district, centers_depots, combinations, adjacent_nodes, average_customers, average_demand,average_workload,shortest_paths_dict, construction_obj,local_infeasible, len(unassigned)
