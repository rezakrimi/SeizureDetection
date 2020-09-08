import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def stack_energy(data):
    x = np.array(data[:, 0:40])
    # x = (x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    energy = x[:, 0:30]
    energy = (energy - energy.mean(axis=0, keepdims=True)) / (energy.max(axis=0, keepdims=True) - energy.min(axis=0, keepdims=True))
    plv = x[41:, 30:40]
    plv = (plv - plv.mean(axis=0, keepdims=True)) / (plv.max(axis=0, keepdims=True) - plv.min(axis=0, keepdims=True))
    t = np.array([i*2 for i in range(energy.shape[0])])
    t_plv = np.array([i * 2 + 41 for i in range(plv.shape[0])])
    energy_stack_number = np.array([i for i in range(energy.shape[1])])
    plv_stack_number = np.array([i for i in range(plv.shape[1])])
    maximum = np.max(plv)
    plt.pcolormesh(t_plv, plv_stack_number, plv.T, vmax=maximum)
    plt.show()




data = sio.loadmat('./patient20_ma_40w_2s_1channel.mat')['data']
stack_energy(data)