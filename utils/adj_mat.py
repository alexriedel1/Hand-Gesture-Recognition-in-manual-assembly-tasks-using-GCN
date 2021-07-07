import numpy as np
from config import CFG

edge_index = np.array([[0, 0, 0,  0,  0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9,  9, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19, 20],
                       [1, 5, 9, 13, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 0, 10,  9, 11, 10, 12, 11,  0, 14, 13, 15, 14, 16, 15,  0, 18, 17, 19, 18, 20, 19]])

#add feats 1
j_from = [4, 5, 8, 9, 12, 13, 16, 17, 13, 20]
j_to =   [5, 4, 9, 8, 13, 12, 17, 16, 20, 13]

#add feats 2 
j2_from = [2, 4, 6, 8, 10, 12, 16, 14, 20, 18]
j2_to =   [4, 2, 8, 6, 12, 10, 14, 16, 18, 20]

adj_dict = {
    0: [1, 5, 9, 13, 17],
    1: [0,2],
    2: [1,3],
    3: [2,4],
    4: [3],
    5: [0,6,9],
    6: [5,7],
    7: [6,8],
    8: [7],
    9: [0, 5,10,13],
    10: [9,11],
    11: [10,12],
    12: [11],
    13: [0, 9,14,17],
    14: [13,15],
    15: [14,16],
    16: [15],
    17: [0,13],
    18: [17,19],
    19: [18,20],
    20: [19],
}

if CFG.add_joints1:
    for f, t in zip(j_from,j_to):
        adj_dict[f].append(t)

if CFG.add_joints2:
    for f, t in zip(j2_from,j2_to):
        adj_dict[f].append(t)

num_node = 21
self_link = [(i, i) for i in range(num_node)]

inward = [  (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        (1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
        (7, 8), (9, 10), (10,11), (11, 12), (13, 14), (14, 15), 
        (15, 16), (17, 18), (18, 19), (19, 20)]

if CFG.add_joints_mode == "rev":
    inward_add1 = [(5,4), (9,8), (13,12), (17,16), (20,13)] #reversed
    inward_add2 = [(4,3), (8,6), (12,10), (16,14), (20,18)] #reversed

if CFG.add_joints_mode == "ori":
    inward_add1 = [(4,5), (8,9), (12,13), (16,17), (13,20)]
    inward_add2 = [(2,4), (6,8), (10,12), (14,16), (18,20)]

if CFG.add_joints1:
    inward += inward_add1

if CFG.add_joints2:
    inward += inward_add2

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

def adj_dict2adj_mat(adj_dict):
    keys=sorted(adj_dict.keys())
    size=len(keys)
    adj_mat = [ [0]*size for i in range(size) ]

    for a,b in [(keys.index(a), keys.index(b)) for a, row in adj_dict.items() for b in row]:
        adj_mat[a][b] = 2 if (a==b) else 1
    return np.array(adj_mat)

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_adj_mat():
    return adj_dict2adj_mat(adj_dict)

def get_norm_adj_mat():
    adj_mat = adj_dict2adj_mat(adj_dict)
    adj_mat_norm = get_normalized_adj(adj_mat)
    return adj_mat_norm

def get_edge_idx():
    return edge_index


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

if __name__ == "__main__":
    adj_mat = get_norm_adj_mat()
    print(adj_mat)