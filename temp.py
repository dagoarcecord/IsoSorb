import numpy as np
import matplotlib.pyplot as plt

def langmuir(Ce, qmax, KL):
    return (qmax * KL * Ce) / (1 + KL * Ce)

print("IsoSorb initialized")