# Perform DeepKriging on all five simulated processes with different samples size
# The RBF centers are [324, 361, 400, 441, 484]
# Will use 7 layers with 200 neurons in each layer
# %% Lib and sys
from Code.central import (MLP, RBFNet, DataModule,
                          Trainer, read_data, get_wkdir, Kriging)
import numpy as np
import torch

wk_dir = get_wkdir()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# %% Process 1: Low variance Matern
coords_full, y_full = read_data("Matern_02_1_001")
iters = 100
lr = 0.1
batch = 64
max_epochs = 125
train_percent = 0.8
centers = [i ** 2 for i in range(18, 23)]
Ns = [2000, 4000, 6000, 8000]
dl_mse = np.zeros((len(Ns), iters))
kr_mse = np.zeros((len(Ns), iters))
for idx, N in enumerate(Ns):
    n_train = int(N * train_percent)
    n_val = N - n_train
    for i in range(iters):
        coords = coords_full[0:N, :, i]
        y = y_full[0:N, i]
        model_rbf = RBFNet(centers, 7, 200, 1, device)
        optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
        data = DataModule(coords, y, n_train, n_val, batch, device)
        trainer = Trainer(max_epochs=max_epochs, device=device)
        train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
        # All validation set error
        y_val = y[n_train:n_train + n_val]
        x_val = coords[n_train:n_train + n_val]
        c = torch.tensor(x_val, device=device).to(torch.float32)
        with torch.no_grad():
            y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
        val_error = np.mean(np.square(y_hat - y_val))
        dl_mse[idx, i] = val_error

        x_train = coords[0:n_train]
        y_train = y[0:n_train]
        mu, hat = Kriging(x_train, x_val, y_train, 0.2)
        kr_err = np.mean(np.square(mu - y_val))
        kr_mse[idx, i] = kr_err

np.savetxt(wk_dir + "Output/dl_Matern_02_1_001.csv", dl_mse, delimiter=",")
np.savetxt(wk_dir + "Output/kr_Matern_02_1_001.csv", kr_mse, delimiter=",")










