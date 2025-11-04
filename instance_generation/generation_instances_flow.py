from dataclasses import dataclass
import random
from random import choice
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import queue
import sys
import codecs
import heapq
import os

nb_instances = 100
nb_instances_generated = 0
#Unit of flow
U = 1000
#Minimal amount of flow to add
min_fl = 1
#A number for print
nb_it_print = 1000

@dataclass
class NodeLF:
    source_id: int
    target_id: int
    value: float
    proba: float

@dataclass
class NodeRF:
    source_id: int
    target_id: int
    value: float

@dataclass
class MATRICE:
    arc: int
    cap: float
    time: float

class Matrix:
    def __init__(self, arc, cap, time):
        self.arc = arc
        self.cap = cap
        self.time = time



def copy_graph(M,V):
    M_= Matrix([[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)])
    for i in range (0, V):
        for j in range (0, V):
            M_.arc[i][j] = M.arc[i][j]
            M_.cap[i][j] = M.cap[i][j]
            M_.time[i][j] = M.time[i][j]
    return M_


def random_pair(LF):
    pp=[]
    for lf in LF:
        pp.append(lf.proba)
    n = list(range(0,len(LF)))
    #print('n',n)
    temp = random.choices(population=n,weights=pp, k=1)
    return temp[0]


def init_cost(s,path_costs,l):
    #print('Initialization of path_costs')
    for j in range(1,l):
        #print(j)
        path_costs[j] = math.inf
    path_costs[s] = 0


def search_min(not_processed, path_costs):
    #print('Initialization')
    cur_min = path_costs[next(iter(not_processed))]
    cur_arg_min = next(iter(not_processed))
    #print('Search the node with minimal estimate of cost value in not_processed')
    for node in not_processed:
        if path_costs[node] < cur_min:
            cur_min = path_costs[node]
            cur_arg_min = node
    return cur_arg_min


def digkstra(M,s,VM,V):
    #print('Initialization of the digkstra cost')
    path_costs = [math.inf for x in range(V)]
    init_cost(s,path_costs,VM)
    #print('Initialization of processed and not_processed')
    processed = set([])
    not_processed = set([])
    for j in range(1,VM):
        not_processed.add(j)


    #print('Main loop')
    while bool(not_processed):
        #print('Search unprocessed node with minimal estimate of cost')
        u = search_min(not_processed,path_costs)
        #print('Update not_processed and processed')
        not_processed.remove(u)
        processed.add(u)
        #print('Relax all edges (u,v) adjacent to u')
        for v in range(1,VM):
            #print('Update the estimates of the edges (u,v) adjacent to u')
            if M.arc[u][v] == 1 and path_costs[v] > path_costs[u] + M.time[u][v]:
                path_costs[v] = path_costs[u] + M.time[u][v]

    return path_costs


def init_DAGSP(DAG,visited,VM):
    for i in range(1,VM):
        visited = False
        for j in range(1,VM):
            DAG[i,j] = 0

def backward_travers_DAGSP_creation(DAG,s,t,M,path_costs,visitied,VM):
    #print('Return NULL if there is no path from s to t')
    if path_costs[t] == math.inf:
        return None

    #print('Queue initialization')
    q = queue.Queue()
    q.put(t)

    while not q.empty():
        #print('Dequeue a node v')
        v = q.get()
        #print('Check the entering neighbours of v')
        for u in range(1,VM):
            if M.arc[u][v] == 1:
                #print('Pick the edge (u,v) in DAG if there is a shortest path from the source to v through u')
                if path_costs[u] + M.time[u][v] == path_costs[v]:
                    DAG[u][v] = 1
                #print('If u is not visited and different from s, mark it as visited and enqueue it for later exploration')
                    if visitied[u] == False:
                        visitied[u] = True
                        if u != s:
                            q.put(u)
    return DAG


def DAGSP(s,t,M,path_costs,V,VM):
    #print('Initialization DAGSP visited')
    visited = [False for i in range(V)]
    DAG = [[0 for x in range(V)]for y in range(V)]
    #init_DAGSP(DAG,visited)
    #print('Graph traversal starting at t')
    DAG = backward_travers_DAGSP_creation(DAG, s, t, M, path_costs, visited,VM)
    return DAG


def return_out_neighbors(M_,u,VM):
    N = set([])
    for v in range (1,VM):
        if M_[u][v] == 1:
            N = N.union({v})
    return N


def random_path(s, t, DAG , VM):
    u = s
    P = set([])

    while u != t:
        out_neighbors = return_out_neighbors(DAG, u, VM)
        #print('out_neighbors', out_neighbors)
        v = choice(list(out_neighbors))
        P = P.union({(u,v)})
        u = v

    return P


def update_time(M,OG,u,v):
    pass

def multiflow(nb_instances_generated):
    #capacity,min_cut,degree
    source_file = "degree"

    file_path = "instances/" + source_file + "/instance_"+str(nb_instances_generated)+".npy"

    instance = np.load(file_path, allow_pickle = True).flat[0]

    adj_mat = instance["adj_mat"]
    capacities = instance["capacities"]
    transport_times = instance["transport_times"]
    pairs = instance["pairs"]
    weight_pairs = instance["weight_pairs"]

    V = len(adj_mat)
    VM = V

    LF = []
    RF = []

    for (s,t) in pairs:
        source_id = s
        target_id = t
        value = 0
        proba = 0
        LF.append(NodeLF(int(source_id),int(target_id),float(value),float(proba)))
        RF.append(NodeRF(int(source_id),int(target_id),float(value)))

    for i in range (0, len(LF)):
        LF[i].value = weight_pairs[i]
        LF[i].proba = LF[i].value/sum(weight_pairs)
    for n in LF:
        n.value = U*n.value
        if n.proba == 'nan':
            n.proba = 0

    for n in RF:
        n.value = 0

    k = len(LF)



    M = Matrix([[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)])

    for i in range(0,V):
        for j in range(0,V):
            M.arc[i][j] = adj_mat[i][j]
            M.cap[i][j] = capacities[i][j]
            M.time[i][j] = transport_times[i][j]



    filename = source_file + 'instance_'+ str(nb_instances_generated)



    cpt1 = 0
    cpt2 = 0
    cpt_saturated = 0
    nb_it = 0
    MF = [[[0 for x in range(V)]for y in range(V)]for z in range(k)]

    #print('copy_graph')
    OG = copy_graph(M,V)

    while cpt_saturated < k:
        #print('Select a pair(si,ti)')
        i = random_pair(LF)

        si = LF[i].source_id
        ti = LF[i].target_id
        #print('Return the DAG of shortest paths')
        path_costs = digkstra(M, si,VM, V)
        DAG = DAGSP(si, ti, M, path_costs,V,VM)
        #print(DAG)
        if DAG:
            #print('Return a random path from the DAG')
            P = random_path(si, ti, DAG, VM)
            capacity = min(M.cap[u][v] for (u,v) in P)
            f_am = min(capacity,LF[i].value,min_fl)

            #print('Increase flow along a path from si to ti and do the relevent updates')
            for (u,v) in P:

                MF[i][u][v] = MF[i][u][v]+1
                M.cap[u][v]= M.cap[u][v]-1
                if M.cap[u][v] == 0:
                    M.arc[u][v] = 0
                else:
                    update_time(M, OG, u, v)

            #print('Decrease the flow value in LF and add it in RF')
            LF[i].value = LF[i].value - 1
            RF[i].value = RF[i].value + 1

            #print('Update the probability of the current pair and increment cpt_saturated')
            if LF[i].value == 0:
                LF[i].proba = 0
                cpt_saturated = cpt_saturated + 1
                cpt1 = cpt1 +1
        else:
            #print('If the DAG is NULL, si and ti are disconnected, do the relevent update')
            LF[i].proba = 0
            cpt_saturated = cpt_saturated + 1
            cpt2 = cpt2 +1

        #if nb_it % nb_it_print == 0:
            #print(nb_it, cpt_saturated,sum(e.value for e in RF), sum(l.value for l in LF),cpt1,cpt2)

        nb_it = nb_it + 1

    flow = [[0 for x in range(V)]for y in range(V)]
    for l in range(0, V):
        for c in range(0, V):
            flow[l][c]=sum(MF[i][l][c] for i in range(0, k))

    nb_arc_flow = sum(flow[l][c]>0 for l in range(0, V) for c in range(0, V))

    tmp = []
    for l in range(0, V):
        for c in range(0, V):
            tmp.append((l,c,flow[l][c], OG.cap[l][c]))
    #nbm = heapq.nlargest(10,tmp,key=lambda x:x[2])

    #print(nbf)
    nb_arc_plein = 0
    for n in tmp:
        #print(n,n[2]/n[3])
        if n[2]>0:
            if n[3] == 0:
                print(n)
            elif n[2]/n[3]== 1:
                #print(n)
                nb_arc_plein = nb_arc_plein+1

    nb_arc_total = sum(M.arc[l][c]>0 for l in range(0, V) for c in range(0, V))


    print(nb_it,sum(e.value for e in RF),nb_arc_flow,nb_arc_plein,nb_arc_total, nb_arc_flow/nb_arc_total, nb_arc_plein/nb_arc_total)

    """with open(filename_Mi, "w", encoding="utf-8") as f, codecs.open(
        f"{filename_Mi}_names", "w", "utf-8"
    ) as f_names:

        for n in MF:
            f.write(f"{n}\n")

    f.close()
    f_names.close()

    with open(filename_RF, "w", encoding="utf-8") as f, codecs.open(
        f"{filename_RF}_names", "w", "utf-8"
    ) as f_names:

        for n in RF:
            f.write(f"{n}\n")

    f.close()
    f_names.close()
    #M = copy_graph(OG)"""

    listRF = []
    for rf in RF:
        listRF.append((rf.source_id, rf.target_id, rf.value))

    flow_and_matrice = {'matrice' : MF, 'flow' : listRF}

    dir_name = "flow_and_matrice/"
    np.save(os.path.join(dir_name, filename), flow_and_matrice)
    #dir_name_MI = "MInpy/"
    #np.save(os.path.join(dir_name_MI, filename_Mi),  MF)

while nb_instances:
    multiflow(nb_instances_generated)

    nb_instances_generated = nb_instances_generated+1
    nb_instances = nb_instances - 1
