import numpy as np
import os

CWD = os.getcwd()
print(CWD)

noisydata = np.load(f'{CWD}/model_results/1.npy', allow_pickle=True)
print(noisydata.shape)