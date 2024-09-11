# Common functions
# %% Import
import os
from logging import raiseExceptions
import matplotlib as mpl
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.special import gamma, kv
import scipy.stats as stats
import platform
import math
import pickle
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
mpl.style.use('ggplot')

# %% System Config
def get_wkdir():
    """
    Get the full working directory based on system platform
    :return: the full working directory
    """
    # Get the working platform
    plat = platform.system()
    if plat == "Linux":
        #return "/share/bjreich/hyang23/SpatialDL/"
        return "/r/bb04na2a.unx.sas.com/vol/bigdisk/lax/hoyang/DLTest/SpatialDL/"
    elif plat == "Darwin":
        return "/Users/hongjianyang/PycharmProjects/SpatialDL/"
    else:
        raise Exception("Working system not defined")

wk_dir = get_wkdir()
def read_data(process):
    """
    Read the low variance matern process
    :param process: The spatial process
    # possible so far: "Matern_02_1_001", "Matern_02_1_09", "Process3", "Process5", and "Process8"
    :return: X and Y of the coresponding spatial process
    """
    assert type(process) == str
    data = pickle.load(open(wk_dir + "Data/" + process, "rb"))
    X = data[:, 0:2, :]
    Y = data[:, 2, :]
    return X, Y

# %% Modules
class Trainer:
    """ Training NN with data """
    def __init__(self, max_epochs, device):
        self.max_epochs = max_epochs
        self.device = device

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model.to(device=self.device)

    def fit(self, model, data, optim):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = optim
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        train_loss = []
        val_loss = []
        for self.epoch in range(self.max_epochs):
            sub_train, sub_val = self.fit_epoch()
            train_loss += sub_train
            val_loss += sub_val
        return train_loss, val_loss

    def fit_epoch(self):
        train_loss = []
        val_loss = []
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(batch)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.train_batch_idx += 1
            train_loss.append(loss.item())
            # print(f'Training batch: {self.train_batch_idx}; Current loss: {loss}')
        if self.val_dataloader.dataset.x.shape[0] == 0:
            return train_loss, val_loss
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                validation_loss = self.model.validation_step(batch)
                val_loss.append(validation_loss.item())
            self.val_batch_idx += 1
        return train_loss, val_loss

class MLP(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size):
        assert num_layers >= 1  # At least one hidden layer
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.LazyLinear(hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.LazyLinear(output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)
        return x

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def training_step(self, batch):
        l = self.loss(self(batch[0]).squeeze(), batch[1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(batch[0]).squeeze(), batch[1])
        return l

class RBF(nn.Module):
    def __init__(self, n_centers, device, grad = False):
        """
        RBF Centers
        :param n_centers: a list of number of centers in each resolution level
        :param grad: True or fase. Whether centers needs gradients
        """
        super().__init__()
        self.out_features = np.sum(n_centers)
        self.n_centers = torch.tensor(n_centers).to(torch.float32)
        self.knots_1d = [torch.linspace(0,1,int(np.sqrt(i))) for i in n_centers]
        self.centers = torch.zeros((self.out_features, 2), device=device)
        sum = 0
        for i in range(len(n_centers)):
            amount = n_centers[i]
            knots_s1, knots_s2 = torch.meshgrid(self.knots_1d[i], self.knots_1d[i])
            knots = torch.column_stack((knots_s1.flatten(), knots_s2.flatten()))
            self.centers[sum:sum + amount, :] = knots
            sum += amount
        self.centers.requires_grad = grad

    def forward(self, x):
        size = (x.size(0), self.out_features, x.size(1))
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.norm(x - centers, dim=2)
        # Applying the specified formula
        rbf_output = (1 - distances) ** 6 * (35 * distances ** 2 + 18 * distances + 3) / 3
        return rbf_output



    # def kernel(self, v1, v2):
    #     d = torch.linalg.norm(v1 - v2)
    #     if 0 <= d <= 1:
    #         return ((1 - d) ** 6) * (35 * torch.square(d) + 18 * d + 3) / 3
    #     else:
    #         return 0
    #
    # def distance_matmul(self, coords):
    #     assert coords.shape[1] == self.center.shape[0] == 2
    #     spaces = 1.0 / torch.tensor(self.n_centers)
    #     result = torch.zeros((coords.shape[0], self.center.shape[1]), device=self.device)
    #     for i in range(coords.shape[0]):
    #         # Keep track of previous index
    #         count = 0
    #         for j in range(len(self.n_centers)):
    #             previous = count
    #             for k in range(int(self.n_centers[j])):
    #                 idx = k + previous
    #                 result[i, idx] = self.kernel(coords[i, :], self.center[:, idx]) / spaces[j]
    #                 count += 1
    #     return result

class RBFNet(MLP):
    def __init__(self, n_centers, num_layers, hidden_size, output_size, device, center_grad = False):
        super().__init__(num_layers, hidden_size, output_size)
        self.center_grad = center_grad
        self.rbf = RBF(n_centers, device, grad=self.center_grad)
        self.rbf = self.rbf.to(device)
        self.initial_run = True

    def forward(self, x):
        x = self.rbf(x)
        return super().forward(x)


# %% Data Module
class SpatialDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DataModule:
    """The base class of data."""
    def __init__(self, x, y, n_train, n_val, batch_size, device):
        self.x = torch.tensor(x.astype(np.float32)).to(device)
        self.y = torch.tensor(y.astype(np.float32)).to(device)
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size

    def get_dataloader(self, train):
        i = slice(0, self.n_train) if train else slice(self.n_train, self.n_train + self.n_val)
        return torch.utils.data.DataLoader(SpatialDataset(self.x[i], self.y[i]), self.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

# %% Common functions
def loss_plot(train_loss, val_loss, title = "Loesses"):
    """
    Plot training and validation errors
    :param train_loss:
    :param val_loss:
    :return:
    """
    train = np.array(train_loss)
    val = np.array(val_loss)
    repeat = int(len(train) / len(val))
    val = np.repeat(val, repeat)
    x_axis = range(len(train))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train, label='Train Loss')
    plt.plot(x_axis, val, label='Val Loss')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# %% Data generation functions
# The process comes from Valid Model Free Prediction paper
# Sklearn implementation of matern
def cor_matern_sk(nu, rho, distance):
    K = distance.copy()
    zero_mask = K == 0
    K[zero_mask] += np.finfo(float).eps  # strict zeros result in nan
    tmp = math.sqrt(8 * nu) / rho * K
    K.fill((2 ** (1.0 - nu)) / gamma(nu))
    K *= tmp ** nu
    K *= kv(nu, tmp)
    return K

def cor_matern(nu, rho, distance):
    """
    The function to calculate the Matern correlation matrix defined in the SPDE paper
    :param nu: smoothness
    :param rho:
    :param distance:
    :return: matern correlation
    """
    kappa = math.sqrt(8 * nu) / rho
    const = (2 ** (1.0 - nu)) / gamma(nu)
    kd = kappa * distance
    first_term = kd**nu
    second_term = kv(nu, kd)
    np.fill_diagonal(second_term, 0.)
    out = const * first_term * second_term
    out[np.diag_indices_from(out)] = 1.0
    return out

def gen_latent(n, nu, rho, spatial_var, length=1):
    """
    Generate the latent Gaussian stationary process without error term
    :param n: Number of observations
    :param nu: smoothness
    :param rho: spatial correlation
    :param spatial_var: spatial variance
    :param noise_var: noise variance
    :param length: the length of the square; Default to 1
    :return: X: spatial coordinates; Y: response
    """
    coords1 = np.random.uniform(0, length, n)
    coords2 = np.random.uniform(0, length, n)
    coords = np.vstack((coords1, coords2)).T
    # Exponential Correlation
    distance = distance_matrix(coords, coords)
    corr = cor_matern(nu, rho, distance)
    # Cholesky decomposition and generate correlated data
    L = np.linalg.cholesky(spatial_var * corr)
    z = np.random.normal(0, 1, n)
    Y = np.dot(L, z)
    return coords, Y

def matern_process(n, nu, rho, spatial_var, noise_std, length=1):
    """
    Generate a 2-D stationary Matern process
    :param n: Number of observations
    :param rho: spatial correlation
    :param spatial_var: spatial variance
    :param noise_var: noise variance
    :param length: the length of the square; Default to 1
    :return: X: spatial coordinates; Y: response
    """
    coords, z = gen_latent(n, nu, rho, spatial_var, length)
    Y = z + np.random.normal(0, noise_std, n)
    return coords, Y

# Process 3: skewed data
# Y (s) = q[Φ{Z(s)/ 3}] + E(s)
def process3(n, nu, rho, spatial_var, noise_std, length=1):
    coords, z = gen_latent(n, nu, rho, spatial_var, length)
    Y = (stats.gamma.ppf(stats.norm.cdf(z / np.sqrt(3)), 1, loc=0, scale=1 / (3**0.5))
         + np.random.normal(0, noise_std, n))
    return coords, Y

# Process 5: nonstatinoary in spatial variance
# Y(s) = sign(Z(s))*|Z(s)|^(s_x+1) + E(s)
def process5(n, nu, rho, spatial_var, noise_std, length=1):
    coords, z = gen_latent(n, nu, rho, spatial_var, length)
    Y = (np.sign(z) * np.abs(z)**(coords[:, 0] + 1)
         + np.random.normal(0, noise_std, n))
    return coords, Y

# Process 8: nonstationary in mean
# Y(s) = Z(S) + 10 * exp(-50||s-c||^2), c = (0.5, 0.5)
def process8(n, nu, rho, spatial_var, noise_std, length=1):
    coords, z = gen_latent(n, nu, rho, spatial_var, length)
    center = np.array([0.5, 0.5]).reshape(-1, 2)
    dist = coords - center
    norm = np.square(np.linalg.norm(dist, ord = 2))
    extra = 10 * np.exp(-50 * norm)
    Y = z + extra + np.random.normal(0, noise_std, n)
    return coords, Y


def Kriging(X_train, X_test, y_train, spatial_corr):
    N = X_train.shape[0] + X_test.shape[0]
    n2 = X_test.shape[0] # Test size
    nu = 1
    s_train = X_train
    s_test = X_test
    coords = np.concatenate((s_test, s_train))
    distance = distance_matrix(coords.reshape(-1, 2), coords.reshape(-1, 2))
    v_bar = np.var(y_train)
    cov_bar = v_bar * cor_matern(nu, spatial_corr, distance)

    sigma11 = cov_bar[0:n2, 0:n2]
    sigma12 = cov_bar[0:n2, n2:N]
    sigma21 = np.transpose(sigma12)
    sigma22 = cov_bar[n2:N, n2:N]

    sigma22_inv = np.linalg.inv(sigma22)
    sigma_bar = sigma11 - np.dot(np.dot(sigma12, sigma22_inv), sigma21)

    mu_bar = np.dot(np.dot(sigma12, sigma22_inv), y_train)

    zzz = np.random.normal(0, 1, n2)
    L = np.linalg.cholesky(sigma_bar)
    y_hat = mu_bar + np.dot(L, zzz)
    return mu_bar, y_hat


