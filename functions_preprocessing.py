#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from windrose import WindroseAxes
from scipy.stats import circvar
from sklearn.cluster import KMeans

# ================ CREATE WEATHER CLUSTERS ==================
# ===========================================================

# 1. gather weather info for a given node
def weather_at_node(folder, node):
    for name in os.listdir(folder): # note name should be a string of format '{int}_'
        if name.startswith('weather_') and node in name: # find the file containing the correct region code
            weather_data = os.path.join(folder, name)
            break
    df_weather = pd.read_csv(weather_data, usecols=['time', 'temp', 'rhum', 'wdir', 'wspd'], index_col='time')
    return df_weather

# 2. given a weather dataset, find the optimal (circvar minimising) k for clustering
def tune_k(df_weather, k_max):
    min_max_circ_var = float('inf')
    optimal_k = 1
    
    # Iterate over a range of k values
    for k in range(1,k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(df_weather)
        labels = kmeans.labels_
        
        # Calculate the maximum circular variance across clusters
        max_circ_var = 0
        for i in range(k):
            cluster_data = df_weather.loc[labels == i, 'wdir'] # get all observations within a cluster
            circvar_value = circvar(np.radians(cluster_data)) # calc the circular variance of that cluster
            if circvar_value > max_circ_var:
                max_circ_var = circvar_value
        
        # Update the optimal k if the current k gives a smaller maximum circular variance
        if max_circ_var < min_max_circ_var:
            min_max_circ_var = max_circ_var
            optimal_k = k
    
    return optimal_k

# 3. given the tuned k, get the resulting clusters of weather
def weather_cluster(df_weather, k): #df_weather is the dataframe containing weather info
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_weather)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    df_cluster = pd.DataFrame(centroids, columns=['temp', 'rhum', 'wdir', 'wspd']) # table detailing the mean temp, RH, wdir and wspd of each cluster
    
    counts = pd.Series(labels).value_counts().sort_index() # number of observations in each cluster
    df_cluster['count'] = counts.values

    df_cluster['circular_var'] = np.nan
    for i in range(k):
        cluster_data = df_weather.loc[labels == i, 'wdir'] # get all observations within a cluster
        circvar_value = circvar(np.radians(cluster_data)) # calc the circular variance of that cluster
        df_cluster.iloc[i,-1] = circvar_value
    
    return df_cluster

# ================= GET ALL SPATIAL INFO (V,E,E_DIRECTED) ===================
# ===========================================================================

# 1. create a dictionary of nodes (V), its location (center_dict), its corners (corner_dict) 
# and the border it shares with other nodes (E + frontier)
def get_node_frontier(node_filename, corner_filename, frontier_threshold): 
    # Load the dictionary of node's vertices coordinates from the txt file
    with open(corner_filename, 'r') as file:
        corner_data = file.read()
    corner_dict = eval(corner_data, {"array": np.array}) # Convert the string representation of the dictionary to an actual dictionary

    # Load a dictionary of node's center coordinates
    node_df = pd.read_csv(node_filename)
    V = [node_df.loc[i,'node_id'] for i in node_df.index]
    init_age = {node_df.loc[i,'node_id']: node_df.loc[i,'initial_age'] for i in node_df.index}
    center_dict = {node_df.loc[i,'node_id']: (node_df.loc[i,'latitude'],node_df.loc[i,'longitude']) for i in node_df.index}

    # Create list of undirected edges based on whether a given pair of nodes share common vertices
    E = []
    frontier = {}

    for i, j in itertools.combinations(V, 2): #combination means notwithstanding order, ie (1,2) is the same as (2,1)
        common = []
        for c_i in corner_dict[i]:
            for c_j in corner_dict[j]:
                if np.linalg.norm(c_i-c_j) <= frontier_threshold:
                    common.append(c_i)
        if common:
            E.append((i,j))
            frontier[(i, j)] = {
                'node1_center': np.array(center_dict[i]),
                'node2_center': np.array(center_dict[j]),
                'frontier_corners': common
            }
    frontier = dict(sorted(frontier.items(), key=lambda item: item[0]))
    
    return corner_dict, center_dict, frontier, V, E, init_age

# 2. given the dominant wind angle, get the unit wind vector
def get_wind_vector_from_angle(angle, scale=1.):
    rad = math.radians(angle)
    return (-scale * math.cos(rad), -scale * math.sin(rad))

# 3. get the cosine value of the angle between 2 given vectors
def get_cos_angle(vector1, vector2):
    dot_prod = np.dot(vector1, vector2)
    mag1 = np.sqrt(np.dot(vector1, vector1))
    mag2 = np.sqrt(np.dot(vector2, vector2))
    return dot_prod / (mag1 * mag2)

# 4. check if vector v lies between vectors a and b
def check_lincom(a,b,v):
    mod = gp.Model('Linear_combo')
    mod.Params.LogToConsole = 0    
    mod.Params.OutputFlag = 0
    
    lambd = mod.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS)
    for i in range(len(a)):
        mod.addConstr(lambd * a[i]/np.linalg.norm(a) + (1-lambd) * b[i]/np.linalg.norm(b) == v[i])
    mod.setObjective(0)
    mod.update()
    mod.optimize()
    if mod.Status == 3:
        lincom = 0
    else:
        lincom = 1
    return lincom

# 5. check if there's a directed edge between (i,j) 
def check_dedge(corner1, corner2, center, wdir, wspd, wind_threshold): 
    #check if wind vector is between the 2 vectors (corner1-center) and (corner2-center)
    a = corner1 - center
    b = corner2 - center
    v = get_wind_vector_from_angle(wdir) # vectorised wind
    lincom = check_lincom(a,b,v)
    
    if lincom == 1: # if wind vector is a linear combo of (corner1-center) and (corner2-center)
        dedge = 1 
    else: # wind vector falls out of the range [(corner1-center); (corner2-center)]
        # project wind vector onto said range's borders
        proj1 = get_cos_angle(v,a)*wspd
        proj2 = get_cos_angle(v,b)*wspd
        proj = max(proj1,proj2)
        # check if projection is above threshold
        if proj >= wind_threshold:
            dedge = 1
        else:
            dedge = 0
    return dedge

# 6. get the wind conditions applicable to a given node based on the weather station closest to it
def get_wdir(folder,V,E,k_max,frontier): # get wdir and wspd for each node
    for i in V:
        node = f'_{i}_'
        df_weather_i = weather_at_node(folder, node) # get weather data relevant to node i
        df_weather_i_filt = df_weather_i.loc[
            (df_weather_i['rhum'] <= np.quantile(df_weather_i.rhum.values, 0.25)) & 
            (df_weather_i['wspd'] >= np.quantile(df_weather_i.wspd.values, 0.25)) &
            (df_weather_i['temp'] >= np.quantile(df_weather_i.temp.values, 0.25))
        ] # filter to most severe fire weather
        
        k = tune_k(df_weather_i_filt, k_max)
        df_weather_i_clust = weather_cluster(df_weather_i_filt, k) # weather clustering 
        wdir = df_weather_i_clust.loc[df_weather_i_clust['count'].idxmax()]['wdir'] # get relevant wind direction
        wspd = df_weather_i_clust.loc[df_weather_i_clust['count'].idxmax()]['wspd'] # get relevant wind speed

        for (i_prime,j_prime) in E:
            if i == i_prime:
                frontier[(i_prime,j_prime)]['node1_wdir'] = wdir
                frontier[(i_prime,j_prime)]['node1_wspd'] = wspd
            elif i == j_prime:
                frontier[(i_prime,j_prime)]['node2_wdir'] = wdir
                frontier[(i_prime,j_prime)]['node2_wspd'] = wspd
    return frontier

# 7. get all the unique wind conditions in the dataset
def get_unique_wdirs(frontier):
    unique_wdirs = set()
    for edge_data in frontier.values():
        if 'node1_wdir' in edge_data:
            unique_wdirs.add(round(edge_data['node1_wdir'], 5))
        if 'node2_wdir' in edge_data:
            unique_wdirs.add(round(edge_data['node2_wdir'], 5))
    unique_wdirs = sorted(list(unique_wdirs))
    return unique_wdirs

def get_unique_wspd(frontier):
    unique_wspd = set()
    for edge_data in frontier.values():
        if 'node1_wspd' in edge_data:
            unique_wspd.add(round(edge_data['node1_wspd'], 5))
        if 'node2_wspd' in edge_data:
            unique_wspd.add(round(edge_data['node2_wspd'], 5))
    unique_wspd = sorted(list(unique_wspd))
    return unique_wspd

# 8. given undirected edges E and the wind vectors for each node, get the set of all directed edges
def get_dedge(E, frontier, wind_threshold): 
    E_dir = []

    for (i, j) in E:
        corner1 = frontier[(i,j)]['frontier_corners'][0]
        corner2 = frontier[(i,j)]['frontier_corners'][1]
        
        center_i = frontier[(i,j)]['node1_center']    
        center_j = frontier[(i,j)]['node2_center']
        
        wdir_i = frontier[(i,j)]['node1_wdir']
        wdir_j = frontier[(i,j)]['node2_wdir']
        
        wspd_i = frontier[(i,j)]['node1_wspd']
        wspd_j = frontier[(i,j)]['node2_wspd']
    
        if check_dedge(corner1, corner2, center_i, wdir_i, wspd_i, wind_threshold) == 1 \
        or check_dedge(corner1, corner2, center_i, wdir_i, wspd_i, wind_threshold*0.9) == 1 \
        or check_dedge(corner1, corner2, center_i, wdir_i, wspd_i, wind_threshold*1.1) == 1:
            E_dir.append((i,j))
            
        if check_dedge(corner1, corner2, center_j, wdir_j, wspd_j, wind_threshold) == 1 \
        or check_dedge(corner1, corner2, center_j, wdir_j, wspd_j, wind_threshold*0.9) == 1 \
        or check_dedge(corner1, corner2, center_j, wdir_j, wspd_j, wind_threshold*1.1) == 1:
            E_dir.append((j,i))
            
    E_dir = sorted(E_dir, key=lambda edge: (edge[0], edge[1]))
    return E_dir

# ==================== VISUALISATION ======================
# =========================================================

# 1. draw the landscape with undirected and directed edges
def draw_before_after(V, center_dict, E, E_dir): 
    xmin = min(center_dict[i][0] for i in center_dict)
    xmax = max(center_dict[i][0] for i in center_dict)
    ymin = min(center_dict[i][1] for i in center_dict)
    ymax = max(center_dict[i][1] for i in center_dict)

    norm_loc = {}
    for key, value in center_dict.items():
        norm_loc[key] = [(value[0] - xmin) / (xmax - xmin),
                         (value[1] - ymin) / (ymax - ymin)]

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(V)
    G_dir.add_edges_from(E_dir)

    options = {'pos':norm_loc,
               'with_labels':True,
               'node_color': 'peru',
               'node_size':150,
               'font_color': 'w',
               'font_size':7}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    nx.draw(G,ax=axes[0],**options)
    axes[0].set_title("Undirected System")

    nx.draw(G_dir,ax=axes[1],**options)
    axes[1].set_title("Directed System")
    # plt.savefig('.png')
    plt.show()
    
    return fig

