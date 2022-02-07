import random
import numpy as np
import matplotlib.pyplot as plt
import torch


class BaseGenerator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ref_X: np.ndarray,
        model,
        use_cuda=True,
        random_seed=42,
    ):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        assert X.ndim == 2, "X shape (n_batch, n_feat) ndarray"
        assert ref_X.ndim == 2, "ref_X shape (n_batch, n_feat) ndarray"
        self.X = X
        self.y = y
        self.n_batch = X.shape[0]
        self.n_feat = X.shape[1]
        self.search_dim = self.n_feat
        self.ref_X = ref_X
        self.model = model
        self.device = "cuda" if use_cuda else "cpu"
        self.cf = []
        self.result = []
        self.target = None

    def get_X(self, idx):
        # sample : (1, n_feat)
        sample = self.X[idx][np.newaxis, ...]
        return sample

    def get_y_hat(self, sample):
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        y_hat = self.model(sample.float().to(self.device))
        return y_hat.detach().cpu().numpy()

    def l2_distance(self, a, b, axis):
        return np.sqrt(((a - b) ** 2).sum(axis=axis))

    def l1_distance(self, a, b, axis):
        return np.abs(a - b).sum(axis=axis)

    def feat_importance(self, sample, cf):
        return (sample - cf) ** 2

    def visualize(self, tag):
        plt.bar(range(self.n_feat), np.array(self.result).sum(axis=0))
        plt.savefig(f'result/feat_importance_{tag}.png')
        plt.show()
