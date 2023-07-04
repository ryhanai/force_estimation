# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import time


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    kde = stats.gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_sklearn(x, x_grid, sample_weights, bandwidth=1.0, **kwargs):
    kde_skl = KernelDensity(kernel='gaussian', bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x, sample_weight=sample_weights)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid)
    return np.exp(log_pdf)


class GridForceMap:
    def __init__(self, name):
        assert name == 'seria_basket' or 'konbini_shelf'
        if name == 'seria_basket':
            self.grid = np.mgrid[-0.095:0.095:40j, -0.13:0.13:40j, 0.73:0.92:40j]
            X, Y, Z = self.grid
            self.dV = 0.19 * 0.26 * 0.20 / (40**3)
            positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            self.positions = positions.T  # [number of points, 3]
            self.bandwidth = 0.010
        elif name == 'konbini_shelf':
            self.grid = np.mgrid[-0.3:0.3:120j, -0.4:0.4:160j, 0.73:0.93:40j]
            X, Y, Z = self.grid
            positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            self.positions = positions.T  # [number of points, 3]
            self.bandwidth = 0.012

        # 'sony_box'
        # self.grid = np.mgrid[-0.115:0.115:40j, -0.115:0.115:40j, 0.93:1.16:40j]
        # 'ipad box'

        self.V = np.zeros(self.positions.shape[0])
        self.alpha = 0.8

        self._title = 'force map'
        self._scene_name = name

    def getDensity(self,
                   sample_positions,
                   sample_weights,
                   moving_average=False,
                   return_3d=False):

        if len(sample_weights) > 0:
            # V = kde_scipy(sample_positions, self.positions, bandwidth=0.3)
            # V = stats.gaussian_kde(sample_positions, bw_method=0.3)
            start_kde = time.time()
            V = kde_sklearn(sample_positions, 
                            self.positions, 
                            sample_weights=sample_weights, 
                            bandwidth=self.bandwidth, 
                            atol=1e-2)
            print(f'KDE took: {time.time() - start_kde} [sec]')
        else:
            V = np.zeros(self.positions.shape[0])

        if moving_average:
            self.V = self.alpha * self.V + (1 - self.alpha) * V
            result = self.V
        else:
            result = V

        if return_3d:
            return result.reshape(self.grid[0].shape)
        else:
            return result

    def set_values(self, values):
        if values.ndim == 3:
            v = np.zeros((self.grid[0].shape))
            v[:, :, :values.shape[-1]] = values
            self.V = v.reshape(self.V.shape)
        else:
            self.V = values

    def get_values(self):
        return self.V

    def get_positions(self):
        return self.positions

    def set_title(self, title):
        self._title = title

    def visualize(self, max_channels=20, zaxis_first=False):
        V = np.reshape(self.V, self.grid[0].shape)
        f = V / np.max(V)
        fig = plt.figure(figsize=(16, 6))
        fig.subplots_adjust(hspace=0.1)
        fig.suptitle(self._title, fontsize=28)

        if zaxis_first:
            channels = f.shape[0]
        else:
            channels = f.shape[-1]
        for p in range(min(channels, max_channels)):
            ax = fig.add_subplot(channels//10, 10, p+1)
            ax.axis('off')
            if zaxis_first:
                ax.imshow(f[p], cmap='gray', vmin=0, vmax=1.0)
            else:
                ax.imshow(f[:, :, p], cmap='gray', vmin=0, vmax=1.0)

    def get_scene(self):
        return self._scene_name

def plot_force_map(force_map, env='seria_basket', title=''):
    fmap = GridForceMap(env)
    fmap.set_values(force_map)
    fmap.set_title(title)
    fmap.visualize()
