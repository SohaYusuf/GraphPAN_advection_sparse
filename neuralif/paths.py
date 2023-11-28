import numpy as np

def ran_symmetric_path():
    path = "C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/Random_sym/"
    return path

def ran_non_symmetric_path(n):
    print(n)
    path = f"C:/Users/soha9/Documents/python/neuralif/neuralif_advection/neuralif/apps/data/Random_{n}"
    return path

def mfem_advection_path():
    path = "C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data_advection/"
    # if mode=="train":
    #     path = "C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/advection/train/cfl=1.387/pt"
    # elif mode=="val":
    #     path = "C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/advection/val/cfl=1.387/pt"
    # elif mode=="test":
    #     path = "C:/Users/soha9/Documents/python/neuralif/neuralif_advection/data/advection/test/cfl=1.387/pt"
    return path