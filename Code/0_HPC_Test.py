import time
tic = time.time()
from central import RBFNet, DataModule, Trainer, read_data, get_wkdir
import numpy as np
import torch
wk_dir = get_wkdir()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Import time: {time.time() - tic}')
print(device)
tic = time.time()
N = 500
coords_full, y_full = read_data("Matern_02_1_001")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]
print(f'Data reading and processing time: {time.time() - tic}')