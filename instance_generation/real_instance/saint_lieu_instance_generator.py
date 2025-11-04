from dataclasses import dataclass
import codecs
import sys


@dataclass
class NodeLF:
    source_id: int
    target_id: int
    value: float
    proba: float

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


def generate_instance_saint_lieu():
    filename_M = 'instance_generation/real_instance/matrixM.pypgr'
    filename_LF = 'instance_generation/real_instance/LF.pypgr'
    #fichier d'origine
    #filename= 'M.pypgr'
    #fichier modifié

    listM = []
    LF = []

    with open(filename_LF, "r", encoding="utf-8") as input_file:
        for line in input_file:
            source_id, target_id, value, proba = line.split(" ")
            LF.append(NodeLF(int(source_id), int(target_id), float(value),float(proba)))

    with open(filename_M, "r", encoding="utf-8") as input_file:
        for line in input_file:
            arc, cap, time = line.split(" ")
            listM.append(MATRICE(int(arc),float(cap),float(time)))

    V = 223
    M = Matrix([[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)],[[0 for x in range(V)]for y in range(V)])

    for i in range (0, V):
        for j in range (0, V):
                M.arc[i][j] = listM[i*V+j].arc
                M.cap[i][j] = listM[i*V+j].cap
                M.time[i][j] = listM[i*V+j].time

    pairs, flow_vals = [], []
    for n in LF:
        #print(n.source_id, n.target_id, n.value, n.proba)
        pairs.append((n.source_id - 1, n.target_id - 1))
        flow_vals.append(n.value)

    adj_mat = [[0 for j in range(V)] for i in range(V)]
    capacities = [[0 for j in range(V)] for i in range(V)]
    transport_times = [[float("inf") for j in range(V)] for i in range(V)]
    for i in range (0, V):
        for j in range (0, V):
            #print(M.arc[i][j],M.cap[i][j],M.time[i][j])
            adj_mat[i][j] = M.arc[i][j]
            capacities[i][j] = M.cap[i][j]
            transport_times[i][j] = M.time[i][j]
    
    return adj_mat, capacities, transport_times, pairs, flow_vals
