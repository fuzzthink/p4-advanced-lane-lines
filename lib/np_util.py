import numpy as np

def scale_255(M):
    return np.uint8(255*M/np.max(M))