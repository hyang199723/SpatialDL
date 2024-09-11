# %%
from Code.central import get_wkdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
wk_dir = get_wkdir()

Ns = [2000, 4000, 6000, 8000]
dl_matern_09 = pd.read_csv(wk_dir + "Output/dl_Matern_02_1_09.csv", header=None)
kr_matern_09 = pd.read_csv(wk_dir + "Output/kr_Matern_02_1_09.csv", header=None)

dl_09_mean = np.mean(dl_matern_09, axis=1)
kr_09_mean = np.mean(kr_matern_09, axis=1)

plt.plot(Ns, dl_09_mean, label='Deep Kriging')
plt.plot(Ns, kr_09_mean, label='Kriging')
plt.legend()
plt.show()
