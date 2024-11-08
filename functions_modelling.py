#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Packages
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import math
import itertools

# ================ 1. ADDITIONAL PREPROCESSING ==================
# ===============================================================

# 1.1 given the initial age, tau0, get the initial fuel load 
def init_load(Linit, Lmax, kappa, tau0):
    x1 = Linit+(Lmax-Linit)*(1-np.exp(-kappa*tau0))
    return x1

# ================ 2. OPTIMISATION MODELLING ===================
# ==============================================================

# 2.1. proposed objective 1 - excess_i + excess_j when i is active
def mod_base(file,V,E,init_age,T,tmin,Linit,Lmax,kappa,alpha,gamma,delta,eta,cost,budget,gap,timelimit):
    mod = gp.Model('Base PB')
    if gap > 0:
        mod.setParam('MIPGap', gap)
    if timelimit > 0:
        mod.setParam('Timelimit', timelimit)
        
    name = file.split('_')[0]
    mod_filename = f"mod_{name}_bg{budget[1]}_T{T}_base.lp" 
    log_filename = f"log_{name}_bg{budget[1]}_T{T}_base.log"

    x = mod.addVars(V, range(1,T+2), lb=0, ub = Lmax, vtype = GRB.CONTINUOUS, name='x')
    h = mod.addVars(V, range(1,T+2), lb=0, ub = Lmax - delta, vtype = GRB.CONTINUOUS, name='h')
    e = mod.addVars(V, range(1,T+2), vtype = GRB.BINARY, name='active')
    z = mod.addVars(V, range(1,T+1), vtype = GRB.BINARY, name='z')

    for i in V:
        mod.addConstr(x[i,1] == init_load(Linit, Lmax, kappa[i], init_age[i]), name = 'initial_load')

        for t in range(1,T+1): # for t in [T]
            mod.addConstr(x[i,t+1] >= alpha[i]*x[i,t], name = 'fuel_if_treat')
            if gamma[i] <= alpha[i]:
                mod.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-gamma[i])*Lmax*z[i,t], name = 'fuel_not_treat1')
            else:
                mod.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-alpha[i])*Lmax*z[i,t], name = 'fuel_not_treat2')
            mod.addConstr(h[i,t] >= x[i,t] - delta, name = 'surplus')
            mod.addConstr(x[i,t] <= delta + Lmax*e[i,t], name = 'activeness1')
            mod.addConstr(x[i,t] >= delta - Lmax*(1-e[i,t]), name = 'activeness2')

        mod.addConstr(h[i,T+1] >= x[i, T+1] - delta, name = 'surplus')
        mod.addConstr(x[i,T+1] <= delta + Lmax*e[i,T+1], name = 'activeness1')
        mod.addConstr(x[i,T+1] >= delta - Lmax*(1-e[i,T+1]), name = 'activeness2')

        if eta[i] >= 1:
            for t in range(1, min(eta[i] + 1, T + 1)):
                mod.addConstr(z[i,t] <= 0, name = 'min_age')

        if tmin[i] + eta[i] <= T-1:
            for t in range(eta[i]+1,T-tmin[i]+1):
                mod.addConstr(gp.quicksum(z[i,s] for s in range(t, t+tmin[i]+1)) <= 1, name = 'min_tfi')
                
    for t in range(1,T+1):
        mod.addConstr(gp.quicksum(z[i,t]*cost[i-1,t-1] for i in V) <= budget[t], name = 'budget')
    
    obj = gp.quicksum(h[i,t] for t in range(1,T+2) for i in V) + gp.quicksum(e[i,t]*h[j,t] for t in range(1,T+2) for (i,j) in E)

    mod.setObjective(obj, sense = GRB.MINIMIZE)

    mod.update()
    mod.write(mod_filename)
    mod.setParam('LogFile', log_filename)
    mod.optimize()
    
    xstar = np.zeros((len(V), T+1))
    hstar = np.zeros((len(V), T+1))
    zstar = np.zeros((len(V), T))
    for i in V:
        for t in range(1,T+1):
            xstar[i-1,t-1] = x[i,t].x
            hstar[i-1,t-1] = h[i,t].x
            zstar[i-1,t-1] = z[i,t].x
        xstar[i-1,T] = x[i,T+1].x
        hstar[i-1,T] = h[i,T+1].x
    
    return xstar, hstar, zstar, mod

# 2.2. proposed objective 2 - largest single-ignition-point connected component
def mod_base_alt(file,V,E,init_age,T,tmin,Linit,Lmax,kappa,alpha,gamma,delta,eta,cost,budget,gap,timelimit):
    mod1 = gp.Model('AltPB')
    if gap > 0:
        mod1.setParam('MIPGap', gap)
    if timelimit > 0:
        mod1.setParam('Timelimit', timelimit)
        
    name = file.split('_')[0]
    mod_filename = f"mod_{name}_bg{budget[1]}_T{T}_alt.lp" 
    log_filename = f"log_{name}_bg{budget[1]}_T{T}_alt.log"

    V2 = list(itertools.product(V, V))

    x = mod1.addVars(V, range(1,T+2), lb=0, ub = Lmax, vtype = GRB.CONTINUOUS, name='x')
    h = mod1.addVars(V, range(1,T+2), lb=0, ub = Lmax-delta, vtype = GRB.CONTINUOUS, name='h')
    a = mod1.addVars(V, range(1,T+2), vtype = GRB.BINARY, name='active') #node activeness
    e = mod1.addVars(set(E), range(1,T+2), lb=0, ub = 1, vtype = GRB.CONTINUOUS, name='edge') #edge activeness
    z = mod1.addVars(V, range(1,T+1), vtype = GRB.BINARY, name='z')
    phi = mod1.addVars(V2, range(1,T+2), lb=0, ub = 1, vtype = GRB.CONTINUOUS, name='concom') #connected component
    prod_phi_h = mod1.addVars(V2, range(1,T+2), lb=0, ub = Lmax-delta, vtype = GRB.CONTINUOUS, name='prod') #prod between phi and h
    sigma = mod1.addVars(range(1,T+2), lb=0, vtype=GRB.CONTINUOUS)
    
    for i in V:
        mod1.addConstr(x[i,1] == init_load(Linit, Lmax, kappa[i], init_age[i]), name = 'initial_load')

        for t in range(1,T+1): # for t in [T]
            mod1.addConstr(x[i,t+1] >= alpha[i]*x[i,t], name = 'fuel_if_treat')
            if gamma[i] <= alpha[i]:
                mod1.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-gamma[i])*Lmax*z[i,t], name = 'fuel_not_treat1')
            else:
                mod1.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-alpha[i])*Lmax*z[i,t], name = 'fuel_not_treat2')
            mod1.addConstr(h[i,t] >= x[i,t] - delta, name = 'surplus')
            mod1.addConstr(x[i,t] <= delta + Lmax*a[i,t], name = 'activeness1')
            mod1.addConstr(x[i,t] >= delta - Lmax*(1-a[i,t]), name = 'activeness2')

        mod1.addConstr(h[i,T+1] >= x[i, T+1] - delta, name = 'surplus')
        mod1.addConstr(x[i,T+1] <= delta + (Lmax - delta)*a[i,T+1], name = 'activeness1')
        mod1.addConstr(x[i,T+1] >= delta*a[i,T+1], name = 'activeness2')

        if eta[i] >= 1:
            for t in range(1, eta[i]+1):
                mod1.addConstr(z[i,t] <= 0, name = 'min_age')

        if tmin[i] + eta[i] <= T-1:
            for t in range(eta[i]+1,T-tmin[i]+1):
                mod1.addConstr(gp.quicksum(z[i,s] for s in range(t, t+tmin[i]+1)) <= 1, name = 'min_tfi')
    
    for (i, j) in E:
        for t in range(1,T+2):
            mod1.addConstr(e[i,j,t] >= a[i,t] + a[j,t] - 1, name=f'edge1_{i}_{j}_{t}')
            mod1.addConstr(e[i,j,t] <= a[i,t], name=f'edge2_{i}_{j}_{t}')
            mod1.addConstr(e[i,j,t] <= a[j,t], name=f'edge3_{i}_{j}_{t}')
            mod1.addConstr(e[i,j,t] <= phi[i,j,t], name=f'concom_{i}_{j}_{t}')
            
    for (i,j) in V2:
        for t in range(1, T+2):
            mod1.addConstr(prod_phi_h[i,j,t] <= h[j,t])
            mod1.addConstr(prod_phi_h[i,j,t] <= phi[i,j,t]*(Lmax-delta))
            mod1.addConstr(prod_phi_h[i,j,t] >= h[j,t] - (1-phi[i,j,t])*(Lmax-delta))
            
            for (i_prime,k) in E:  # Use a different name for the first index in the inner loop to avoid confusion
                if i == i_prime and k != j:
                    mod1.addConstr(e[i,k,t] - 1 <= phi[i,j,t] - phi[k,j,t], name=f'phi_constr1_{i}_{j}_{k}_{t}')
                    mod1.addConstr(phi[i,j,t] - phi[k,j,t] <= 1 - e[i,k,t], name=f'phi_constr2_{i}_{j}_{k}_{t}')

#             if i < j:
#                 mod1.addConstr(phi[i,j,t] == phi[j,i,t])
                
    for t in range(1,T+1):
        mod1.addConstr(gp.quicksum(z[i,t]*cost[i-1,t-1] for i in V) <= budget[t], name = 'budget')

    for t in range(1,T+2):    
        for i in V:
            mod1.addConstr(sigma[t] >= gp.quicksum(prod_phi_h[i,j,t] for j in V))

    mod1.setObjective(gp.quicksum(sigma[t] for t in range(1,T+2)), sense = GRB.MINIMIZE)

    mod1.update()
    mod1.write(mod_filename)
    mod1.setParam('LogFile', log_filename)
    mod1.optimize()
    
    xstar = np.zeros((len(V), T+1))
    hstar = np.zeros((len(V), T+1))
    zstar = np.zeros((len(V), T))
    for i in V:
        for t in range(1,T+1):
            xstar[i-1,t-1] = x[i,t].x
            hstar[i-1,t-1] = h[i,t].x
            zstar[i-1,t-1] = z[i,t].x 

        xstar[i-1,T] = x[i,T+1].x
        hstar[i-1,T] = h[i,T+1].x

    return xstar, hstar, zstar, mod1

# 2.3. benchmark 1 - minimise total excess fuel
def mod_comp_load(file,V,init_age,T,tmin,Linit,Lmax,kappa,alpha,gamma,delta,eta,cost,budget,gap,timelimit):
    mod = gp.Model('SumPB')
    if gap > 0:
        mod.setParam('MIPGap', gap)
    if timelimit > 0:
        mod.setParam('Timelimit', timelimit)
        
    name = file.split('_')[0]
    mod_filename = f"mod_{name}_bg{budget[1]}_T{T}_load.lp" 
    log_filename = f"log_{name}_bg{budget[1]}_T{T}_load.log"

    x = mod.addVars(V, range(1,T+2), lb=0, vtype = GRB.CONTINUOUS, name='x')
    h = mod.addVars(V, range(1,T+2), lb=0, vtype = GRB.CONTINUOUS, name='h')
    z = mod.addVars(V, range(1,T+1), vtype = GRB.BINARY, name='z')

    for i in V:
        mod.addConstr(x[i,1] == init_load(Linit, Lmax, kappa[i], init_age[i]), name = 'initial_load')

        for t in range(1,T+1): # for t in [T]
            mod.addConstr(x[i,t+1] >= alpha[i]*x[i,t], name = 'fuel_if_treat')
            if gamma[i] <= alpha[i]:
                mod.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-gamma[i])*Lmax*z[i,t], name = 'fuel_not_treat1')
            else:
                mod.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-alpha[i])*Lmax*z[i,t], name = 'fuel_not_treat2')
            mod.addConstr(h[i,t] >= x[i,t] - delta, name = 'surplus')

        mod.addConstr(h[i,T+1] >= x[i, T+1] - delta, name = 'surplus')

        if eta[i] >= 1:
            for t in range(1, eta[i]+1):
                mod.addConstr(z[i,t] <= 0, name = 'min_age')

        if tmin[i] + eta[i] <= T-1:
            for t in range(eta[i]+1,T-tmin[i]+1):
                mod.addConstr(gp.quicksum(z[i,s] for s in range(t, t+tmin[i]+1)) <= 1, name = 'min_tfi')
                
    for t in range(1,T+1):
        mod.addConstr(gp.quicksum(z[i,t]*cost[i-1,t-1] for i in V) <= budget[t], name = 'budget')
    
    obj = gp.quicksum(h[i,t] for t in range(1,T+2) for i in V)

    mod.setObjective(obj, sense = GRB.MINIMIZE)

    mod.update()
    mod.write(mod_filename)
    mod.setParam('LogFile', log_filename)
    mod.optimize()
    
    xstar = np.zeros((len(V), T+1))
    hstar = np.zeros((len(V), T+1))
    zstar = np.zeros((len(V), T))
    for i in V:
        for t in range(1,T+1):
            xstar[i-1,t-1] = x[i,t].x
            hstar[i-1,t-1] = h[i,t].x
            zstar[i-1,t-1] = z[i,t].x
        xstar[i-1,T] = x[i,T+1].x
        hstar[i-1,T] = h[i,T+1].x
    
    return xstar, hstar, zstar, mod

# 2.4. benchmark 2 - minimise total number of active edges
def mod_comp_conn(file,V,E,init_age,T,tmin,Linit,Lmax,kappa,alpha,gamma,delta,eta,cost,budget,gap,timelimit):
    mod1 = gp.Model('ConnPB')
    if gap > 0:
        mod1.setParam('MIPGap', gap)
    if timelimit > 0:
        mod1.setParam('Timelimit', timelimit)
        
    name = file.split('_')[0]
    mod_filename = f"mod_{name}_bg{budget[1]}_T{T}_conn.lp" 
    log_filename = f"log_{name}_bg{budget[1]}_T{T}_conn.log"

    x = mod1.addVars(V, range(1,T+2), lb=0, ub = Lmax, vtype = GRB.CONTINUOUS, name='x')
    a = mod1.addVars(V, range(1,T+2), vtype = GRB.BINARY, name='active') #node activeness
    e = mod1.addVars(set(E), range(1,T+2), vtype = GRB.BINARY, name='edge') #edge activeness
    z = mod1.addVars(V, range(1,T+1), vtype = GRB.BINARY, name='z')

    for i in V:
        mod1.addConstr(x[i,1] == init_load(Linit, Lmax, kappa[i], init_age[i]), name = 'initial_load')

        for t in range(1,T+1): # for t in [T]
            mod1.addConstr(x[i,t+1] >= alpha[i]*x[i,t], name = 'fuel_if_treat')
            if gamma[i] <= alpha[i]:
                mod1.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-gamma[i])*Lmax*z[i,t], name = 'fuel_not_treat1')
            else:
                mod1.addConstr(x[i,t+1] >= gamma[i]*x[i,t] + (1-gamma[i])*Lmax - (1-alpha[i])*Lmax*z[i,t], name = 'fuel_not_treat2')
            mod1.addConstr(x[i,t] <= delta + Lmax*a[i,t], name = 'activeness1')
            mod1.addConstr(x[i,t] >= delta - Lmax*(1-a[i,t]), name = 'activeness2')

        mod1.addConstr(x[i,T+1] <= delta + Lmax*a[i,T+1], name = 'activeness1')
        mod1.addConstr(x[i,T+1] >= delta - Lmax*(1-a[i,T+1]), name = 'activeness2')

        if eta[i] >= 1:
            for t in range(1, eta[i]+1):
                mod1.addConstr(z[i,t] <= 0, name = 'min_age')

        if tmin[i] + eta[i] <= T-1:
            for t in range(eta[i]+1,T-tmin[i]+1):
                mod1.addConstr(gp.quicksum(z[i,s] for s in range(t, t+tmin[i]+1)) <= 1, name = 'min_tfi')
    
    for (i, j) in E:
        for t in range(1,T+2):
            mod1.addConstr(e[i,j,t] >= a[i,t] + a[j,t] - 1, name=f'edge1_{i}_{j}_{t}')
            mod1.addConstr(e[i,j,t] <= a[i,t], name=f'edge2_{i}_{j}_{t}')
            mod1.addConstr(e[i,j,t] <= a[j,t], name=f'edge3_{i}_{j}_{t}')
                
    for t in range(1,T+1):
        mod1.addConstr(gp.quicksum(z[i,t]*cost[i-1,t-1] for i in V) <= budget[t], name = 'budget')
    
    obj = gp.quicksum(e[i,j,t] for t in range(1,T+2) for (i,j) in E)

    mod1.setObjective(obj, sense = GRB.MINIMIZE)

    mod1.update()
    mod1.write(mod_filename)
    mod1.setParam('LogFile', log_filename)
    mod1.optimize()
    
    xstar = np.zeros((len(V), T+1))
    zstar = np.zeros((len(V), T))
    for i in V:
        for t in range(1,T+1):
            xstar[i-1,t-1] = x[i,t].x
            zstar[i-1,t-1] = z[i,t].x
        xstar[i-1,T] = x[i,T+1].x
    hstar = np.maximum(xstar - delta,0)
    
    return xstar, hstar, zstar, mod1

# ================ 3. TREATMENT PLAN INTERPRETATION ===================
# =====================================================================

# 3.1. list treatment decisions for each period
def treatment_plan(zstar,V,T):
    df = pd.DataFrame(index=['Node to treat'], columns=[f't = {t}' for t in range(1, T+1)])
    for t in range(T):
        treated_node = [f'{i+1}' for i in range(len(V)) if zstar[i, t] == 1]
        df.at['Node to treat', f't = {t + 1}'] = ', '.join(treated_node)
    df
    return df

# 3.2. draw the landscape over the planning years
def draw_network(file, V, E, loc, hstar, T, zstar_name):
    surplus_tot = np.zeros((len(V), T + 1))
    for t in range(T + 1):
        for (i, j) in E:
            surplus_tot[i - 1, t] += hstar[j - 1, t]
            surplus_tot[j - 1, t] += hstar[i - 1, t]

    labels = {t: {i: np.round(hstar[i - 1, t - 1], 1) for i in V} for t in range(1, T + 2)}
#     labels = {t: {i: i for i in V} for t in range(1, T + 2)}

    xmin = min(loc[i][0] for i in loc)
    xmax = max(loc[i][0] for i in loc)
    ymin = min(loc[i][1] for i in loc)
    ymax = max(loc[i][1] for i in loc)

    norm_loc = {key: [(value[0] - xmin) / (xmax - xmin),
                      (value[1] - ymin) / (ymax - ymin)]
                for key, value in loc.items()}

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    # Calculate rows and columns for subplots
    cols = 3
    rows = math.ceil((T+1) / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5))
    axs = axs.flatten() if T+1 > 1 else [axs]

    for t in range(1, T + 2):
        ax = axs[t - 1]
        ax.set_title(f'Fuel network t={t}')

        for u, v, d in G.edges(data=True):
            d['weight'] = hstar[u - 1, t - 1] * hstar[v - 1, t - 1]

        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        
        normalized_fuel_loads = (hstar[:, t - 1] - np.min(hstar)) / (np.max(hstar) - np.min(hstar))

        options = {'pos': norm_loc,
                   #'node_size': surplus_tot[:, t - 1] * 20,
                   'node_size': pd.Series([300]*len(V)),
                   'node_color': normalized_fuel_loads * (-1),
                   'cmap': plt.get_cmap("RdYlGn"),
                   'edgelist': edges,
                   'edge_color': weights,
                   'edge_cmap': plt.cm.Greys}
        nx.draw(G, ax=ax, **options)

        label_pos = {node: (x, y) #if surplus_tot[node - 1, t - 1] > 11 else (x - 0.04, y + 0.02)
                     for node, (x, y) in norm_loc.items()}
        
        # Determine font color based on node color
        node_colors = options['node_color']
        color_map = options['cmap']
        norm = plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max())
        node_color_values = color_map(norm(node_colors))

        for node, (x, y) in label_pos.items():
            color = node_color_values[node - 1]  # node - 1 to match the index
            grayscale_value = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            font_color = 'white' if grayscale_value < 0.5 else 'black'
            ax.text(x, y, labels[t][node], fontsize=10, color=font_color, ha='center', va='center')

        ax.set_axis_off()

    # Turn off any excess subplots
    for idx in range(T+1, len(axs)):
        axs[idx].axis('off')
        
    name = file[0].split('_')[0]
    img_name = f"{name}_{zstar_name}"

    plt.tight_layout()
    plt.savefig(f'{img_name}.png')
    plt.show()
    
# ================ 4. TREATMENT PLAN PERFORMANCE ======================
# =====================================================================

# 4.1. given undirected spread, calculate the number of nodes and fuel load burnt in each period
def evalu(V, E, T, h):
    df = pd.DataFrame(index=['Number of nodes burnt', 'Fuel load burnt'], columns=[f't={t}' for t in range(1, T+2)])
    largest_subtrees = {f't={t}': [] for t in range(1, T+2)}

    for t in range(T+1):
        fire_scale = 0
        fire_loss = 0
        visited = set()
        
        for i in V:
            if h[i-1, t] > 0 and i not in visited:
                spread = set()
                nodes_to_visit = [i]
                
                while nodes_to_visit:
                    current_node = nodes_to_visit.pop()
                    if current_node not in spread:
                        spread.add(current_node)
                        visited.add(current_node)
                        for j in V:
                            if (current_node, j) in E or (j, current_node) in E: #undirected case
                                if h[j-1, t] > 0 and j not in spread: 
                                        nodes_to_visit.append(j)
                                    
                current_fire_scale = len(spread)
                current_fire_loss = sum(h[n-1, t] for n in spread)
                
                if current_fire_loss > fire_loss:
                    fire_loss = current_fire_loss
                if current_fire_scale > fire_scale or (current_fire_scale == fire_scale and current_fire_loss > fire_loss):
                    fire_scale = current_fire_scale
                    largest_subtree = spread

        df.loc['Number of nodes burnt', f't={t+1}'] = fire_scale
        df.loc['Fuel load burnt', f't={t+1}'] = np.round(fire_loss,2)
        largest_subtrees[f't={t+1}'] = list(largest_subtree)
    
    return df, largest_subtrees

# 4.2. given directed spread, calculate the number of nodes and fuel load burnt in each period
def eval_dir(V, E, T, h):
    df = pd.DataFrame(index=['Number of nodes burnt', 'Fuel load burnt'], columns=[f't={t}' for t in range(1, T+2)])
    largest_subtrees = {f't={t}': [] for t in range(1, T+2)}

    for t in range(T+1):
        fire_scale = 0
        fire_loss = 0
        visited = set()
        
        for i in V:
            if h[i-1, t] > 0 and i not in visited:
                spread = set()
                nodes_to_visit = [i]
                
                while nodes_to_visit:
                    current_node = nodes_to_visit.pop()
                    if current_node not in spread:
                        spread.add(current_node)
                        visited.add(current_node)
                        for j in V: 
                            if (current_node, j) in E and h[j-1, t] > 0 and j not in spread: #directed case
                                        nodes_to_visit.append(j)
                                    
                current_fire_scale = len(spread)
                current_fire_loss = sum(h[n-1, t] for n in spread)
                
                if current_fire_loss > fire_loss:
                    fire_loss = current_fire_loss
                if current_fire_scale > fire_scale or (current_fire_scale == fire_scale and current_fire_loss > fire_loss):
                    fire_scale = current_fire_scale
                    largest_subtree = spread

        df.loc['Number of nodes burnt', f't={t+1}'] = fire_scale
        df.loc['Fuel load burnt', f't={t+1}'] = np.round(fire_loss,2)
        largest_subtrees[f't={t+1}'] = list(largest_subtree)
    
    return df, largest_subtrees

# 4.3. summarise performance comparison in a table
def perf_table(df, df_alt, df_load, df_conn, T):
    # Calculate average losses
    avg_loss_obj1 = sum(df.iloc[1, 1:]) / T
    avg_loss_obj2 = sum(df_alt.iloc[1, 1:]) / T
    avg_loss_load = sum(df_load.iloc[1, 1:]) / T
    avg_loss_conn = sum(df_conn.iloc[1, 1:]) / T

    # Calculate extra loss compared to OBJ 1
    extra_loss_obj2 = (sum(df_alt.iloc[1, 1:]) - sum(df.iloc[1, 1:])) / T
    extra_loss_load = (sum(df_load.iloc[1, 1:]) - sum(df.iloc[1, 1:])) / T
    extra_loss_conn = (sum(df_conn.iloc[1, 1:]) - sum(df.iloc[1, 1:])) / T

    # Create a DataFrame
    results = pd.DataFrame({
        'Average Loss': [avg_loss_obj1, avg_loss_obj2, avg_loss_load, avg_loss_conn],
        'Extra Loss Compared to Obj 1': [0, extra_loss_obj2, extra_loss_load, extra_loss_conn]
    }, index=['OBJ 1', 'OBJ 2', 'LOAD', 'CONN'])

    return results