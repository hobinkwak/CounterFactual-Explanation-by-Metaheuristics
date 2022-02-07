import numpy as np
import torch
from optim.woa import WOGenerator
from optim.ga import GAGenerator
from model import DNN


def main(*args, mode='woa', visualize=True):
    if mode != 'ga':
        worker = WOGenerator(*args)
    else:
        worker = GAGenerator(*args)
    worker.run(visualize)


if __name__ == '__main__':
    mode= 'woa'
    X = np.load('data/train_x.npy')
    y = np.load('data/train_y.npy')
    dnn = DNN(X.shape[-1], y.shape[-1])
    model = torch.load('model/dnn.pt').to('cuda')
    main(X[:1], y[:1], X, model, mode=mode)
