import numpy as np


def create_cactus_data(solved, time_used, var_count):
    rows = [[x, y, z] for x, y, z in zip(var_count, solved, time_used)]
    rows = sorted(rows, key=lambda x: x[0])
    rows = [[y, z] for _, y, z in rows]
    return np.cumsum(rows, axis=0)
