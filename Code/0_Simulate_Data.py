# Simulate data for future use
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
mpl.style.use('ggplot')
# %% Config
from Code.central import (
    matern_process, get_wkdir, process3, process5, process8)
wk_dir = get_wkdir()

nugget_std = 0.1
rho = 0.2
nu = 1
spatial_var = 1
iters = 100
N = 8000
# %% Simulate Matern process
# Test matern process and plot
coords, Y = matern_process(N, nu, rho, spatial_var, nugget_std)
plt.scatter(coords[:, 0], coords[:, 1], c=Y, s=3)
plt.colorbar()
plt.title("Matern process")
plt.tight_layout()
plt.show()

output = np.zeros((N, 3, iters))
for i in range(iters):
    coords, Y = matern_process(N, nu, rho, spatial_var, nugget_std)
    output[:, 0:2, i] = coords
    output[:, 2, i] = Y
# Dump to pickle file
with open(wk_dir + "Data/Matern_02_1_001", "wb") as f:
    pickle.dump(output, f)

# %% Simulate higher noise variance matern
nugget_std = np.sqrt(0.8) # spatial var : noise var = 1:0.8
coords, Y = matern_process(N, nu, rho, spatial_var, nugget_std)
plt.scatter(coords[:, 0], coords[:, 1], c=Y, s=3)
plt.colorbar()
plt.title("Matern process with higher noise variance (1:0.8)")
plt.tight_layout()
plt.show()

output = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    coords, Y = matern_process(N, nu, rho, spatial_var, nugget_std)
    output[:, 0:2, i] = coords
    output[:, 2, i] = Y
# Dump to pickle file
with open(wk_dir + "Data/Matern_02_1_09", "wb") as f:
    pickle.dump(output, f)

# %% Process 3: Y (s) = Z(s)^3 + E(s)
nugget_std = 0.01
coords, Y3 = process3(N, nu, rho, spatial_var, nugget_std)
plt.scatter(coords[:, 0], coords[:, 1], c=Y3, s=3)
plt.colorbar()
plt.title("P3: Stationary but non-Gaussian (skewed)")
plt.tight_layout()
plt.show()

sns.histplot(Y3, bins=30)
plt.title("Distribution of P3")
plt.show()

output = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    coords, Y3 = process3(N, nu, rho, spatial_var, nugget_std)
    output[:, 0:2, i] = coords
    output[:, 2, i] = Y3
# Dump to pickle file
with open(wk_dir + "Data/Process3", "wb") as f:
    pickle.dump(output, f)

# %% Process 5: Y(s) = sign(Z(s))*|Z(s)|^(s_x+1) + E(s)
# nonstatinoary in spatial variance
nugget_std = 0.01
coords, Y5 = process5(N, nu, rho, spatial_var, nugget_std)
plt.scatter(coords[:, 0], coords[:, 1], c=Y5, s=3)
plt.colorbar()
plt.title("P5: Non-stationary in spatial variance")
plt.tight_layout()
plt.show()

output = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    coords, Y5 = process5(N, nu, rho, spatial_var, nugget_std)
    output[:, 0:2, i] = coords
    output[:, 2, i] = Y5
# Dump to pickle file
with open(wk_dir + "Data/Process5", "wb") as f:
    pickle.dump(output, f)

# %% Process 8: Process 8: nonstationary in mean
# Y(s) = Z(S) + 10 * exp(-50||s-c||^2), c = (0.5, 0.5)
nugget_std = 0.01
coords, Y8 = process8(N, nu, rho, spatial_var, nugget_std)
plt.scatter(coords[:, 0], coords[:, 1], c=Y8, s=3)
plt.colorbar()
plt.title("P8: Non-stationary in mean")
plt.tight_layout()
plt.show()

output = np.zeros((N, 3, iters))
for i in range(iters):
    print(i)
    coords, Y8 = process8(N, nu, rho, spatial_var, nugget_std)
    output[:, 0:2, i] = coords
    output[:, 2, i] = Y8
# Dump to pickle file
with open(wk_dir + "Data/Process8", "wb") as f:
    pickle.dump(output, f)


