import json
import math
from functools import partial
import numpy as np
import torch
from optim import BaseGenerator

class WOGenerator(BaseGenerator):
    def __init__(self, X, y, ref_X, model, target=None):
        super().__init__(X, y, ref_X, model)
        self.woa_param = json.load(open("config/optim_param.json", "r"))["woa"]
        if target:
            self.target = target
        self.whale = {}
        self.prey = {}

    def get_bounds(self):
        lb = self.ref_X.min(axis=0).flatten().tolist()
        ub = self.ref_X.max(axis=0).flatten().tolist()
        self.lb = lb
        self.ub = ub

    def obj_func(self, data, sample, y_hat):
        data = data.astype('float32')
        cf_x = torch.tensor(data).view(-1, self.n_feat).float()
        x_diff = self.l1_distance(sample, data, axis=-1)
        cf_y_hat = self.get_y_hat(cf_x)
        if self.target is not None:
            y_hat_diff = self.l2_distance(y_hat[:, self.target: self.target + 1],
                                          cf_y_hat[:, self.target: self.target + 1], axis=-1) * (-1)
        else:
            y_hat_diff = self.l2_distance(y_hat, cf_y_hat, axis=-1) * (-1)
        result = x_diff + y_hat_diff
        return result

    def init_whale(self, obj_func):
        tmp = [[np.random.uniform(self.lb[j], self.ub[j]) for j in range(len(self.lb))]
               for i in range(self.woa_param['n_whale'])]
        self.whale['position'] = np.array(tmp)
        self.whale['fitness'] = obj_func(self.whale['position'])

    def init_prey(self, obj_func):
        self.prey['position'] = np.zeros((1, len(self.lb)))
        self.prey['fitness'] = obj_func(self.prey['position'])[0]

    def update_prey(self):
        if self.whale['fitness'].min() < self.prey['fitness']:
            self.prey['position'][0] = self.whale['position'][self.whale['fitness'].argmin()]
            self.prey['fitness'] = self.whale['fitness'].min()

    def search(self, idx, A, C):
        random_whale = self.whale['position'][np.random.randint(low=0, high=self.woa_param['n_whale'],
                                                                size=len(idx[0]))]
        d = np.abs(C[..., np.newaxis] * random_whale - self.whale['position'][idx])
        self.whale['position'][idx] = np.clip(random_whale - A[..., np.newaxis] * d, self.lb, self.ub)

    def encircle(self, idx, A, C):
        d = np.abs(C[..., np.newaxis] * self.prey['position'] - self.whale['position'][idx])
        self.whale['position'][idx] = np.clip(self.prey['position'][0] - A[..., np.newaxis] * d, self.lb, self.ub)

    def bubble_net(self, idx):
        d_prime = np.abs(self.prey['position'] - self.whale['position'][idx])
        l = np.random.uniform(-1, 1)
        self.whale['position'][idx] = np.clip(
            d_prime * np.exp(self.woa_param['spiral'] * l) * math.cos(2 * np.pi * l) + self.prey['position'][0],
            self.lb, self.ub)

    def optimize(self, a, obj_func):

        p = np.random.random(self.woa_param['n_whale'])
        r = np.random.random(self.woa_param['n_whale'])
        A = 2 * a * r - a
        C = 2 * r
        search_idx = np.where((p < 0.5) & (A > 1))
        encircle_idx = np.where((p < 0.5) & (A <= 1))
        bubbleNet_idx = np.where(p >= 0.5)
        self.search(search_idx, A[search_idx], C[search_idx])
        self.encircle(encircle_idx, A[encircle_idx], C[encircle_idx])
        self.bubble_net(bubbleNet_idx)
        self.whale['fitness'] = obj_func(self.whale['position'])

    def run(self, visualize=True):
        self.get_bounds()
        for i in range(self.n_batch):
            sample = self.get_X(i)
            y_hat = self.get_y_hat(sample)
            obj_func = partial(self.obj_func, sample=sample, y_hat=y_hat)
            self.init_whale(obj_func)
            self.init_prey(obj_func)
            for n in range(self.woa_param['n_iter']):
                print("Iteration = ", n, " f(x) = ", self.prey['fitness'])
                a = 2 - n * (2 / self.woa_param['n_iter'])
                self.optimize(a, obj_func)
                self.update_prey()
            f_value, cf = self.prey['fitness'].squeeze(), self.prey['position'].squeeze()
            self.cf.append(cf)
            feat_imp = self.feat_importance(sample.squeeze(), cf)
            self.result.append(feat_imp)
        self.result = np.array(self.result)
        self.cf = np.array(self.cf)
        np.save("result/cf_instance_woa.npy", self.cf)
        np.save("result/feat_importance_woa.npy", self.result)
        if visualize:
            self.visualize('woa')
