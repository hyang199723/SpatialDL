# Dry run on spatial data with latest codes
# %%
import time
tic = time.time()
from Code.central import (MLP, RBFNet, DataModule,
                          Trainer, read_data, get_wkdir, loss_plot, Kriging)
import numpy as np
import torch
wk_dir = get_wkdir()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# %% Process 1
N = 8000
coords_full, y_full = read_data("Matern_02_1_001")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]
# %%
centers = [i ** 2 for i in range(18, 23)]
lr = 0.05
batch = 64
max_epochs = 125
train_percent = 0.8
n_train = int(N * train_percent)
n_val = N - n_train
y_val = y[n_train:n_train + n_val]
x_val = coords[n_train:n_train + n_val]
# Compute Static centers
model_rbf = RBFNet(centers, 7, 200, 1, device)
optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
data = DataModule(coords, y, n_train, n_val, batch, device)
trainer = Trainer(max_epochs=max_epochs, device=device)
tic = time.time()
train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
toc = time.time()
# All validation set error
c = torch.tensor(x_val, device=device).to(torch.float32)
with torch.no_grad():
    y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
val_error = np.mean(np.square(y_hat - y_val))

title = f'P1: {n_train} training and {n_val} validation; Elapsed = {toc - tic:03f} second'
loss_plot(train_loss, val_loss, title)

# %% Kriging
x_train = coords[0:n_train]
y_train = y[0:n_train]
mu, hat = Kriging(x_train, x_val, y_train, 0.2)

# %% Process 2
coords_full, y_full = read_data("Matern_02_1_09")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]

centers = [i ** 2 for i in range(18, 23)]
lr = 0.05
batch = 64
max_epochs = 125
train_percent = 0.8
n_train = int(N * train_percent)
n_val = N - n_train
# Compute Static centers
model_rbf = RBFNet(centers, 7, 200, 1, device)
optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
data = DataModule(coords, y, n_train, n_val, batch, device)
trainer = Trainer(max_epochs=max_epochs, device=device)
tic = time.time()
train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
toc = time.time()
# All validation set error
y_val = y[n_train:n_train + n_val]
x_val = coords[n_train:n_train + n_val]
c = torch.tensor(x_val, device=device).to(torch.float32)
with torch.no_grad():
    y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
val_error = np.mean(np.square(y_hat - y_val))

title = f'P2: {n_train} training and {n_val} validation; Elapsed = {toc - tic:03f} second'
loss_plot(train_loss, val_loss, title)

# %% Process 3
coords_full, y_full = read_data("Process3")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]

centers = [i ** 2 for i in range(18, 23)]
lr = 0.1
batch = 64
max_epochs = 125
train_percent = 0.8
n_train = int(N * train_percent)
n_val = N - n_train
# Compute Static centers
model_rbf = RBFNet(centers, 7, 200, 1, device)
optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
data = DataModule(coords, y, n_train, n_val, batch, device)
trainer = Trainer(max_epochs=max_epochs, device=device)
tic = time.time()
train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
toc = time.time()
# All validation set error
y_val = y[n_train:n_train + n_val]
x_val = coords[n_train:n_train + n_val]
c = torch.tensor(x_val, device=device).to(torch.float32)
with torch.no_grad():
    y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
val_error = np.mean(np.square(y_hat - y_val))

title = f'P3: {n_train} training and {n_val} validation; Elapsed = {toc - tic:03f} second'
loss_plot(train_loss, val_loss, title)

# %% Process 4
coords_full, y_full = read_data("Process5")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]

centers = [i ** 2 for i in range(18, 23)]
lr = 0.1
batch = 64
max_epochs = 125
train_percent = 0.8
n_train = int(N * train_percent)
n_val = N - n_train
# Compute Static centers
model_rbf = RBFNet(centers, 7, 200, 1, device)
optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
data = DataModule(coords, y, n_train, n_val, batch, device)
trainer = Trainer(max_epochs=max_epochs, device=device)
tic = time.time()
train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
toc = time.time()
# All validation set error
y_val = y[n_train:n_train + n_val]
x_val = coords[n_train:n_train + n_val]
c = torch.tensor(x_val, device=device).to(torch.float32)
with torch.no_grad():
    y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
val_error = np.mean(np.square(y_hat - y_val))

title = f'P4: {n_train} training and {n_val} validation; Elapsed = {toc - tic:03f} second'
loss_plot(train_loss, val_loss, title)

# %% Process 5
coords_full, y_full = read_data("Process8")
coords = coords_full[:N, :, 0]
y = y_full[:N, 0]

centers = [i ** 2 for i in range(18, 23)]
lr = 0.1
batch = 64
max_epochs = 125
train_percent = 0.8
n_train = int(N * train_percent)
n_val = N - n_train
# Compute Static centers
model_rbf = RBFNet(centers, 7, 200, 1, device)
optimizer = torch.optim.SGD(model_rbf.parameters(), lr)
data = DataModule(coords, y, n_train, n_val, batch, device)
trainer = Trainer(max_epochs=max_epochs, device=device)
tic = time.time()
train_loss, val_loss = trainer.fit(model_rbf, data, optimizer)
toc = time.time()
# All validation set error
y_val = y[n_train:n_train + n_val]
x_val = coords[n_train:n_train + n_val]
c = torch.tensor(x_val, device=device).to(torch.float32)
with torch.no_grad():
    y_hat = model_rbf(c).squeeze().detach().cpu().numpy()
val_error = np.mean(np.square(y_hat - y_val))

title = f'P5: {n_train} training and {n_val} validation; Elapsed = {toc - tic:03f} second'
loss_plot(train_loss, val_loss, title)



