import numpy as np

data = np.load("multi_flow_generation/real_instances/capacity/capacity0.npy", allow_pickle=True).flat[0]

print(len(data["matrice"]))

print(len(data["flow"]))

print(len(data))