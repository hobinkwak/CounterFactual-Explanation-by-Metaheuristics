import json
import pickle as pkl
from functools import partial
import numpy as np
import torch
from geneticalgorithm import geneticalgorithm
from optim import BaseGenerator


class GAGenerator(BaseGenerator):
    def __init__(self, X, y, ref_X, model):
        super().__init__(X, y, ref_X, model)
        self.ga_param = json.load(open('config/optim_param.json', 'r'))['ga']

    def make_varbound(self):
        max_bound = self.ref_X.max(axis=0).tolist()
        min_bound = self.ref_X.min(axis=0).tolist()
        bounds = list(zip(min_bound, max_bound))
        self.varbound = np.array([list(tup) for tup in bounds])

    def objective_function(self, data, sample, y_hat):
        cf_x = torch.from_numpy(data).view(1, self.n_feat).float()
        x_diff = self.l1_distance(sample, data)
        cf_hat = self.get_y_hat(cf_x)
        y_hat_diff = self.l2_distance(y_hat, cf_hat)
        return x_diff - y_hat_diff

    def optimize(self, sample, y_hat, plot_show=False):
        obj_func = partial(self.objective_function, sample=sample, y_hat=y_hat)
        model_ga = geneticalgorithm(function=obj_func,
                                    dimension=self.n_feat,
                                    variable_type='real',
                                    variable_boundaries=self.varbound,
                                    algorithm_parameters=self.ga_param,
                                    convergence_curve=plot_show)
        model_ga.run()
        return model_ga.output_dict['variable']

    def run(self, visualize=True, plot_show=False):
        self.make_varbound()
        for i in range(self.n_batch):
            sample = self.get_X(i)
            y_hat = self.get_y_hat(sample)
            cf = self.optimize(sample, y_hat, plot_show)
            self.cf.append(cf)
            feat_imp = self.feat_importance(sample.squeeze(), cf)
            self.result.append(feat_imp)
        self.result = np.array(self.result)
        self.cf = np.array(self.cf)
        np.save("result/cf_instance_ga.npy", self.cf)
        np.save("result/feat_importance_ga.npy", self.result)
        if visualize:
            self.visualize('woa')
