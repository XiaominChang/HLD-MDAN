import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import importlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


import sys
import logging
import numpy as np


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def data_loader(inputs, targets, batch_size, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    inputs_size = inputs.shape[0]
    if shuffle:
        random_order = np.arange(inputs_size)
        np.random.shuffle(random_order)
        inputs, targets = inputs[random_order, :], targets[random_order]
    num_blocks = int(inputs_size / batch_size)
    for i in range(num_blocks):
        yield inputs[i * batch_size: (i+1) * batch_size, :], targets[i * batch_size: (i+1) * batch_size]
    if num_blocks * batch_size != inputs_size:
        yield inputs[num_blocks * batch_size:, :], targets[num_blocks * batch_size:]


def multi_data_loader(inputs, targets, batch_size, shuffle=True):
    """
    Both inputs and targets are list of numpy arrays, containing instances and labels from multiple sources.
    """
    assert len(inputs) == len(targets)
    input_sizes = [data.shape[0] for data in inputs]
    max_input_size = max(input_sizes)
    num_domains = len(inputs)
    if shuffle:
        for i in range(num_domains):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i] = inputs[i][r_order, :], targets[i][r_order]
    num_blocks = int(max_input_size / batch_size)
    for j in range(num_blocks):
        xs, ys = [], []
        for i in range(num_domains):
            ridx = np.random.choice(input_sizes[i], batch_size)
            xs.append(inputs[i][ridx, :])
            ys.append(targets[i][ridx])
        yield xs, ys

from sklearn.metrics import  zero_one_loss,mean_absolute_error,r2_score, mean_squared_error
from math import sqrt
def get_mae(target, prediction):
    assert (target.shape == prediction.shape)

    return mean_absolute_error(target, prediction)

def get_mse(target, prediction):
    assert (target.shape == prediction.shape)

    return mean_squared_error(target, prediction)

def get_sae(target, prediction):
    assert (target.shape == prediction.shape)

    r = target.sum()
    r0 = prediction.sum()
    sae = abs(r0 - r) / r
    return sae

def get_nde(target, prediction):
    assert (target.shape == prediction.shape)

    error, squarey = [], []
    for i in range(len(prediction)):
        value = prediction[i] - target[i]
        error.append(value * value)
        squarey.append(target[i] * target[i])
    nde = sqrt(sum(error) / sum(squarey))
    return nde

def dataProvider(file,windowsize,stepsize,threshold):
    dataframe=pd.read_csv(file,header=0)
    np_array=np.array(dataframe)
    inputs, targets=np_array[:, 0], np_array[:, 1]
    offset = int(0.5 * (windowsize - 1.0))
#     window_num=inputs.size - 2 * offset
    features=list()
    labels=list()
    i=0
    while (i<=inputs.size-windowsize):
        data_in=inputs[i:i+windowsize]
        tar=targets[i+offset]
        features.append(data_in)
        labels.append(tar)
        i=i+stepsize
    X=np.array(features)
    Y=np.array(labels)
    Y[Y<=threshold] = 0
    # scaler = MinMaxScaler()
    # X=scaler.fit_transform(X)
    x_train_all, X, y_train_all, Y = train_test_split(X, Y, test_size=0.2,random_state=100)
    return X,Y


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[1, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if i == 0:
            joint_kernels = torch.ones_like(kernels)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss1 += joint_kernels[s1, s2] + joint_kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss2 -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


def JAN_Linear(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[2, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if i == 0:
            joint_kernels = torch.ones_like(kernels)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    # Linear version
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


# loss_dict = {"DAN":DAN, "DAN_Linear":DAN_Linear, "RTN":RTN, "JAN":JAN, "JAN_Linear":JAN_Linear}

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:56:20 2020

@author: MXM
"""

import torch


def kernel(X, X2, gamma=0.4):
    '''
    Input: X  Size1*n_feature
           X2 Size2*n_feature
    Output: Size1*Size2
    '''
    X = torch.transpose(X, 1, 0)
    X2 = torch.transpose(X2, 1, 0)
    n1, n2 = X.shape[1], X2.shape[1]
    n1sq = torch.sum(X ** 2, 0)
    n1sq = n1sq.float().to(device)
    n2sq = torch.sum(X2 ** 2, 0)
    n2sq = n2sq.float().to(device)
    D = torch.ones((n1, n2)).to(device) * n2sq + torch.transpose((torch.ones((n2, n1)).to(device) * n1sq), 1,
                                                                 0) + - 2 * torch.mm(torch.transpose(X, 1, 0), X2)
    K = torch.exp(-gamma * D)
    return K


def MLcon_kernel(X_p, Y_p, X_q, Y_q, lamda=1):
    '''
    dim(X_p_list) = dim(X_q_list) = layer_num*Size*n_feature
    here we set layer_num = 1
    '''
    layer_num = 1
    out = 0
    np = X_p.shape[0]
    nq = X_q.shape[0]
    I1 = torch.eye(np).to(device)
    I2 = torch.eye(nq).to(device)
    Kxpxp = kernel(X_p, X_p)
    Kxqxq = kernel(X_q, X_q)
    Kxqxp = kernel(X_q, X_p)
    Kypyq = kernel(Y_p, Y_q)
    Kyqyq = kernel(Y_q, Y_q)
    Kypyp = kernel(Y_p, Y_p)
    a = torch.mm((torch.inverse(Kxpxp + np * lamda * I1)), Kypyp)
    b = torch.mm(a, (torch.inverse(Kxpxp + np * lamda * I1)))
    c = torch.mm(b, Kxpxp)
    out1 = torch.trace(c)

    a1 = torch.mm((torch.inverse(Kxqxq + nq * lamda * I2)), Kyqyq)
    b1 = torch.mm(a1, (torch.inverse(Kxqxq + nq * lamda * I2)))
    c1 = torch.mm(b1, Kxqxq)
    out2 = torch.trace(c1)

    a2 = torch.mm((torch.inverse(Kxpxp + np * lamda * I1)), Kypyq)
    b2 = torch.mm(a2, (torch.inverse(Kxqxq + nq * lamda * I2)))
    c2 = torch.mm(b2, Kxqxp)
    out3 = torch.trace(c2)
    out += (out1 + out2 - 2 * out3)
    return out


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    # @staticmethod
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


mse = torch.nn.MSELoss()


# def weighted_mse(outputs, target, alpha):
#     """
#     Spectral Norm Loss between one source and one target: ||output^T*output-target^T*target||_2
#     Inputs:
#         - output: torch.Tensor, source distribution
#         - target: torch.Tensor, target distribution
#     Output:
#         - loss: float, value of the spectral norm of the difference of covariance matrix

#     """
#     loss = torch.sum(torch.stack([alpha[i]*torch.sqrt(mse(outputs[i], target[i])) for i in range(len(outputs))]))
#     return loss


def weighted_mse(outputs, target, alpha):
    """
    Spectral Norm Loss between one source and one target: ||output^T*output-target^T*target||_2
    Inputs:
        - output: torch.Tensor, source distribution
        - target: torch.Tensor, target distribution
    Output:
        - loss: float, value of the spectral norm of the difference of covariance matrix

    """
    loss = torch.sum(torch.stack([alpha[i] * mse(outputs[i], target[i]) for i in range(len(outputs))]))
    return loss


def weighted_jmmd(source, target, alpha):
    loss = torch.sum(torch.stack([alpha[i] * JAN_Linear(source[i], target) for i in range(len(source))]))
    return loss


def weighted_ceod(sourceList_x, sourceList_y, target_x, target_y, alpha):
    loss = torch.sum(torch.stack([alpha[i] * MLcon_kernel(sourceList_x[i], sourceList_y[i], target_x, target_y) for i in
                                  range(len(sourceList_x))]))
    return loss


# def weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, alpha):
#     loss = torch.sum(torch.stack([alpha[i]*torch.abs(torch.sqrt(mse(y_spred[i], y_sdisc[i]))-torch.sqrt(mse(y_tpred, y_tdisc[i]))) for i in range(len(y_spred))]))
#     return loss

def weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, alpha):
    loss = torch.sum(torch.stack(
        [alpha[i] * torch.abs(mse(y_spred[i], y_sdisc[i]) - mse(y_tpred, y_tdisc[i])) for i in range(len(y_spred))]))
    return loss


# def weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, alpha):
#     loss = torch.sum(torch.stack([alpha[i]*torch.abs(mse(y_spred[i], y_sdisc[i]) - mse(y_tpred, y_tdisc[i])) for i in range(len(y_spred))]))
#     return loss

class Disc_MSDANet(nn.Module):
    """
    Multi-Source Domain Adaptation with Discrepancy: adapts from multi-source with the hDiscrepancy
    Learns both a feature representation and weight alpha
    params:
        - 'input_dim': input dimension
        - 'hidden_layers': list of number of neurons in each layer
        - 'output_dim': output dimension (1 in general)
    """

    def __init__(self, params):
        super(Disc_MSDANet, self).__init__()
        self.input_dim = params["input_dim"]
        self.output_dim = params['output_dim']
        self.n_sources = params['n_sources']
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.feature_extractor = params['feature_extractor']
        # Parameter of the final regressor.
        self.h_pred = params['h_pred']
        self.h_disc = params['h_disc']
        self.loss = params['loss']
        self.weighted_loss = params['weighted_loss']
        self.min_pred = params['min_pred']
        self.max_pred = params['max_pred']
        # Parameter
        self.register_parameter(name='alpha',
                                param=torch.nn.Parameter(torch.Tensor(np.ones(self.n_sources) / self.n_sources)))
        self.register_parameter(name='mu', param=torch.nn.Parameter(torch.Tensor(np.array([0.9, 0.1]))))
        self.register_parameter(name='beta', param=torch.nn.Parameter(torch.Tensor(np.array([0.6, 0.4]))))
        self.grls = [GradientReversalLayer() for _ in range(self.n_sources)]

    def optimizers(self, opt_feat, opt_pred, opt_disc, opt_alpha, opt_mu, opt_beta):
        """
        Defines optimizers for each parameter
        """
        self.opt_feat = opt_feat
        self.opt_pred = opt_pred
        self.opt_disc = opt_disc
        self.opt_alpha = opt_alpha
        self.opt_mu = opt_mu
        self.opt_beta = opt_beta

    def reset_grad(self):
        """
        Set all gradients to zero
        """
        self.opt_feat.zero_grad()
        self.opt_pred.zero_grad()
        self.opt_disc.zero_grad()
        self.opt_alpha.zero_grad()
        self.opt_mu.zero_grad()
        self.opt_beta.zero_grad()

    def extract_features(self, x):
        z = x.clone()
        for hidden in self.feature_extractor:
            z = hidden(z)
        return z

    def forward(self, X_s, X_t):
        """
        Forward pass
        Inputs:
            - X_s: list of torch.Tensor (m_s, d), source data
            - X_t: torch.Tensor (n, d), target data
        Outputs:
            - y_spred: list of torch.Tensor (m_s), h source prediction
            - y_sdisc: list of torch.Tensor (m_s), h' source prediction
            - y_tpred: list of torch.Tensor (m_s), h target prediction
            - y_tdisc: list of torch.Tensor (m_s), h' target prediction
        """
        # Feature extractor
        sx, tx = X_s.copy(), X_t.clone()
        for i in range(self.n_sources):
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])
        #                 print(sx[i].shape)
        for hidden in self.feature_extractor:
            tx = hidden(tx)

        # Predictor h
        y_spred = []
        for i in range(self.n_sources):
            y_sx = sx[i].clone()
            y_sx = self.h_pred(y_sx)
            y_spred.append(self.clamp(y_sx))

        y_tx = tx.clone()
        y_tx = self.h_pred(y_tx)
        y_tpred = self.clamp(y_tx)

        # Discrepant h'
        y_sdisc, y_tdisc = [], []
        for i in range(self.n_sources):
            y_tmp = sx[i].clone()
            #             y_tmp =self.grls[i].apply(y_tmp)
            y_tmp = self.h_disc[i](y_tmp)
            y_sdisc.append(self.clamp(y_tmp))
            y_tmp = tx.clone()
            #             y_tmp =self.grls[i].apply(y_tmp)
            y_tmp = self.h_disc[i](y_tmp)
            y_tdisc.append(self.clamp(y_tmp))
        return y_spred, y_sdisc, y_tpred, y_tdisc

    def train_prediction(self, X_s, X_t, y_s, clip=1, pred_only=False):
        """
        Train phi and h to minimize the source error
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        - Outputs:
        """
        # Training
        self.train()

        # Prediction training
        self.reset_grad()
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        loss_pred = self.weighted_loss(y_s, y_spred, self.alpha)
        loss_pred.backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        # Optimization step
        self.opt_pred.step()
        if not pred_only:
            self.opt_feat.step()
        self.reset_grad()
        return loss_pred

    def train_feat_discrepancy(self, X_s, X_t, y_s, clip=1, mu=0.5):
        """
        Train phi to minimize the discrepancy
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        """
        # Training
        self.train()

        # Feature training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = torch.abs(self.weighted_loss(y_spred, y_sdisc, self.alpha) - torch.sqrt(self.loss(y_tpred, y_tdisc)))
        source_loss = self.weighted_loss(y_s, y_spred, self.alpha)
        sourceList = []
        for i in range(self.n_sources):
            z_sfeat = self.extract_features(X_s[i])
            sourceList.append([z_sfeat, y_spred[i]])
        z_tfeat = self.extract_features(X_t)
        targetList = [z_tfeat, y_tpred]
        loss_jmmd = weighted_jmmd(sourceList, targetList, self.alpha)
        loss = mu * disc + source_loss + mu * loss_jmmd
        self.reset_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        # Optimization step
        self.opt_feat.step()
        self.reset_grad()
        return loss

    def train_all_pred(self, X_s, X_t, y_s, clip=1):
        self.train()
        lam_alpha = 0.001
        # Prediction training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        source_loss = self.weighted_loss(y_s, y_spred, self.alpha)
        # disc_loss = weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, self.alpha)
        sourceList_x = []
        sourceList_y = []
        for i in range(self.n_sources):
            z_sfeat = self.extract_features(X_s[i])
            sourceList_x.append(z_sfeat)
            sourceList_y.append(y_spred[i])
        z_tfeat = self.extract_features(X_t)
        target_x = z_tfeat
        target_y = y_tpred
        ceod_loss = weighted_ceod(sourceList_x, sourceList_y, target_x, target_y, self.alpha)
        # source_loss=torch.exp(source_loss/1000)/(torch.exp(source_loss/1000)+torch.exp(ceod_loss/1000))
        # ceod_loss=torch.exp(ceod_loss/1000)/(torch.exp(source_loss/1000)+torch.exp(ceod_loss/1000))
        # loss=source_loss+ ceod_loss
        loss = self.mu[0] * source_loss + self.mu[1] * ceod_loss
        self.reset_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        # Optimization step
        self.opt_pred.step()
        self.reset_grad()
        return loss, source_loss, ceod_loss

    def train_all_feat(self, X_s, X_t, y_s, clip=1):
        self.train()
        # Prediction training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        source_loss = self.weighted_loss(y_s, y_spred, self.alpha)
        disc_loss = weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, self.alpha)
        sourceList_x = []
        sourceList_y = []
        for i in range(self.n_sources):
            z_sfeat = self.extract_features(X_s[i])
            sourceList_x.append(z_sfeat)
            sourceList_y.append(y_spred[i])
        z_tfeat = self.extract_features(X_t)
        target_x = z_tfeat
        target_y = y_tpred
        ceod_loss = weighted_ceod(sourceList_x, sourceList_y, target_x, target_y, self.alpha)
        # loss=source_loss+ 0.1*(self.beta[0]*ceod_loss - self.beta[1]* disc_loss)+lam_alpha*torch.norm(self.alpha, p=2)
        # loss=self.mu[0]*source_loss+ self.mu[1]*(self.beta[0]*ceod_loss + self.beta[1]* disc_loss)+lam_alpha*torch.norm(self.alpha, p=2)
        # source_loss=torch.exp(source_loss/1000)/(torch.exp(source_loss/1000)+torch.exp(disc_loss/1000)+torch.exp(ceod_loss/1000))
        # disc_loss=math.exp(disc_loss/1000)/(torch.exp(source_loss/1000)+torch.exp(disc_loss/1000)+torch.exp(ceod_loss/1000))
        # ceod_loss=math.exp(ceod_loss/1000)/(torch.exp(source_loss/1000)+torch.exp(disc_loss/1000)+torch.exp(ceod_loss/1000))
        loss = self.mu[0] * source_loss + self.mu[1] * (self.beta[0] * ceod_loss + self.beta[1] * disc_loss)
        self.reset_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        # Optimization step
        self.opt_feat.step()
        self.opt_mu.step()
        self.opt_beta.step()
        self.reset_grad()
        # Normalization (||alpha||_1=1)
        with torch.no_grad():
            self.mu.clamp_(1 / 100, 1 - 1 / 100)
            self.mu.div_(torch.norm(F.relu(self.mu), p=1))
            self.beta.clamp_(1 / 100, 1 - 1 / 100)
            self.beta.div_(torch.norm(F.relu(self.beta), p=1))
        return loss, source_loss, ceod_loss, disc_loss

    def train_all_alpha(self, X_s, X_t, y_s, clip=1):
        self.train()
        lam_alpha = 0.001
        # Prediction training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc_loss = weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, self.alpha)
        sourceList_x = []
        sourceList_y = []
        for i in range(self.n_sources):
            z_sfeat = self.extract_features(X_s[i])
            sourceList_x.append(z_sfeat)
            sourceList_y.append(y_spred[i])
        z_tfeat = self.extract_features(X_t)
        target_x = z_tfeat
        target_y = y_tpred
        ceod_loss = weighted_ceod(sourceList_x, sourceList_y, target_x, target_y, self.alpha)
        loss = disc_loss + ceod_loss + lam_alpha * torch.norm(self.alpha, p=2)
        self.reset_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        # Optimization step
        self.opt_alpha.step()
        self.reset_grad()
        # Normalization (||alpha||_1=1)
        with torch.no_grad():
            self.alpha.clamp_(1 / (self.n_sources * 10), 1 - 1 / (self.n_sources * 10))
            self.alpha.div_(torch.norm(F.relu(self.alpha), p=1))
        return loss

    def train_disc(self, X_s, X_t, y_s, clip=1):
        self.train()
        # Prediction training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = self.weighted_loss(y_s, y_sdisc, self.alpha)
        disc_loss = -weighted_disc(y_spred, y_sdisc, y_tpred, y_tdisc, self.alpha)
        loss = disc + disc_loss
        self.reset_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        # Optimization step
        self.opt_disc.step()
        self.reset_grad()
        return loss

    def predict(self, X):
        z = X.clone()
        for hidden in self.feature_extractor:
            z = hidden(z)
        #         for hidden in self.h_pred:
        z = self.h_pred(z)
        return self.clamp(z)

    def clamp(self, x):
        return torch.clamp(x, self.min_pred, self.max_pred)

    logger = logging.getLogger(__name__)

class Flatten(torch.nn.Module):
    def forward(self, x):
            batch_size = x.shape[0]
            return x.view(batch_size, -1)

def block(in_feat, out_feat):
        layers = [nn.Linear(in_feat, out_feat)]
        layers.append(nn.BatchNorm1d(out_feat))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

def conv(in_feat, out_feat, filter_size=5, stride=1, padding=True):
        if padding == True:
            p = int((filter_size - 1) / 2)
        else:
            p = 0
        conv_layers = [nn.Conv1d(in_feat, out_feat, filter_size, stride, padding=p)]
        conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
        return conv_layers

def flat():
        flatten = Flatten()
        layers = [flatten]
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

def get_feature_extractor():
        g1 = nn.Sequential(
            *conv(in_feat=1, out_feat=30, filter_size=10, stride=1, padding=True),
        )

        g2 = nn.Sequential(
            *conv(in_feat=30, out_feat=30, filter_size=8, stride=1, padding=True),
        )

        g3 = nn.Sequential(
            *conv(in_feat=30, out_feat=40, filter_size=6, stride=1, padding=True),
        )

        g4 = nn.Sequential(
            *conv(in_feat=40, out_feat=50, filter_size=5, stride=1, padding=True),

        )

        g5 = nn.Sequential(
            *conv(in_feat=50, out_feat=50, filter_size=5, stride=1, padding=True),
            *flat(),
        )

        g6 = nn.Sequential(
            *block(800, 1024),
        )
        extractor = nn.ModuleList([g1, g2, g3, g4, g5, g6])
        return extractor

def get_predictor(output_dim=1):
        return nn.Linear(1024, output_dim)
        #            nn.Sequential(

    #            nn.Linear(1024, output_dim))
    #            nn.Linear(1024, output_dim),nn.LeakyReLU(0,inplace=False))
    #     return  nn.ModuleList([
    #             #nn.Linear(100,10, bias=False), nn.ELU(), nn.Dropout(p=0.1),
    #             nn.Linear(1024, output_dim, bias=False)])

def get_discriminator(num_domains, output_dim=1):
        return nn.ModuleList([
            nn.Linear(1024, output_dim) for _ in range(num_domains)])



trainfile1="redd_training/fridge_house_3_training_.csv"
trainfile2="redd_training/fridge_house_2_training_.csv"
trainfile3="ukdale_training/fridge_house_1_training_.csv"
trainfile4="ukdale_training/fridge_house_2_training_.csv"
trainfile5="refit_training/fridge/fridge_house_12_training_.csv"
trainfile6="refit_training/fridge/fridge_house_15_training_.csv"
fileList=[trainfile1,trainfile2,trainfile3,trainfile4,trainfile5,trainfile6]
stepsizeList=[2,2,16,16,16,16]
data_insts, data_labels, num_insts,on = [], [], [],[]
threshold=50
for i in range(len(fileList)):
    X,Y=dataProvider(fileList[i], 19, stepsizeList[i], threshold)
    Y=Y.reshape(-1,1)
    data_insts.append(X)
    data_labels.append(Y)
    num_insts.append(X.shape[0])
min_size=min(num_insts)
x_tem, y_tem=[],[]
for j in range(len(fileList)):
  ridx = np.random.choice(num_insts[j], min_size)
  x_tem.append(data_insts[j][ridx, :])
  y_tem.append(data_labels[j][ridx, :])
  num_insts[j]=x_tem[j].shape[0]
data_insts=x_tem
data_labels=y_tem
print(num_insts)

num_data_sets = len(fileList)
num_domains = num_data_sets - 1
nb_experiments = 1

# -np.inf
params = {'input_dim': 19, 'output_dim': 1, 'n_sources': num_data_sets - 1, 'loss': torch.nn.MSELoss(),
          'weighted_loss': weighted_mse, 'min_pred': 0, 'max_pred': np.inf}

# Number of epochs
epochs_pretrain, epochs_adapt = 50, 20

for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------' % (exp + 1, nb_experiments))
    mse_list, mae_list = {}, {}
    alphas = {}
    for i in [0, 3, 5]:
        print(
            '\n---------------------------------------------- domain num %i is running----------------------------------' % (
                i))
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j])
                source_labels.append(data_labels[j])
        # Build target instances.
        target_idx = i
        target_insts = data_insts[i]
        target_labels = data_labels[i]
        target_insts0, X_t, target_labels0, y_t = train_test_split(target_insts, target_labels, test_size=0.2,
                                                                   random_state=100)
        X_t = X_t.reshape(X_t.shape[0], 1, X_t.shape[1])
        X_t = torch.tensor(X_t, requires_grad=False).type(torch.FloatTensor).to(device)

        X = np.concatenate(source_insts, axis=0)
        Y = np.concatenate(source_labels, axis=0)

        x_train_all, x_test, y_train_all, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        x_test = torch.tensor(x_test, requires_grad=False).type(torch.FloatTensor).to(device)

        ############################# Build and define model ####################################
        params['feature_extractor'] = get_feature_extractor()
        params['h_pred'] = get_predictor(output_dim=1)
        params['h_disc'] = get_discriminator(num_domains, output_dim=1)
        model = Disc_MSDANet(params).to(device)

        #         #Pre-training
        print('------------Pre-training------------')
        lr = 0.0001
        lr1 = 0.00001
        batch_size = 64
        b1 = 0.99
        b2 = 0.999
        #         opt_feat=torch.optim.SGD([{'params': model.feature_extractor.parameters()}], lr=0.0001, momentum=0.8)
        #         opt_pred=torch.optim.SGD([{'params': model.h_pred.parameters()}], lr=0.0001, momentum=0.8)
        opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}], lr=lr, betas=(b1, b2))
        opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}], lr=lr, betas=(b1, b2))
        opt_disc = torch.optim.Adam([{'params': model.h_disc.parameters()}], lr=lr, betas=(b1, b2))
        opt_alpha = torch.optim.Adam([{'params': model.alpha}], lr=lr, betas=(b1, b2))
        opt_mu = torch.optim.Adam([{'params': model.mu}], lr=lr, betas=(b1, b2))
        opt_beta = torch.optim.Adam([{'params': model.beta}], lr=lr, betas=(b1, b2))
        model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha, opt_mu, opt_beta)
        mae_loss, mse_loss = [], []
        for epoch in range(epochs_pretrain):
            running_loss = 0.0
            model.train()
            #             input_sizes = [data.shape[0] for data in source_insts]
            #             max_input_size = max(input_sizes)
            #             K=int(max_input_size/batch_size)
            #             print('K value is:', K)

            loader = multi_data_loader(source_insts, source_labels, batch_size)

            for x_bs, y_bs in loader:
                for j in range(num_domains):
                    x_bs[j] = x_bs[j].reshape(x_bs[j].shape[0], 1, x_bs[j].shape[1])
                    x_bs[j] = torch.tensor(x_bs[j], requires_grad=False).type(torch.FloatTensor).to(device)
                    y_bs[j] = torch.tensor(y_bs[j], requires_grad=False).type(torch.FloatTensor).to(device)
                ridx = np.random.choice(target_insts.shape[0], batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = tinputs.reshape(tinputs.shape[0], 1, tinputs.shape[1])
                x_t = torch.tensor(tinputs, requires_grad=False).type(torch.FloatTensor).to(device)
                loss_pred = model.train_prediction(x_bs, x_t, y_bs, clip=1, pred_only=False)
                #                 print("predictor is:", model.h_pred.weight.data)
                running_loss += loss_pred
            print('Epoch: %i; MSE loss: %.3f' % (epoch, running_loss))
            model.eval()
            prediction = model.predict(X_t).cpu().detach().numpy()
            stopT = get_mae(y_t, prediction)
            mae_loss.append(stopT)
            print(stopT)
            cur_loss = get_mse(y_t, prediction)
            mse_loss.append(cur_loss)
            print(cur_loss)
            if (epoch + 1) % 10 == 0:
                model.eval()
                prediction = model.predict(X_t).cpu().detach().numpy()
                #                 source_loss, disc = model.compute_loss(x_bs, X_t, y_s)
                reg_loss = get_mae(y_t, prediction)
                reg_mse = get_mse(y_t, prediction)
                print('Epoch: %i; Test MAE loss on target data is: %.3f' % (epoch + 1, reg_loss))
                print('Epoch: %i; Test MSE loss on target data is: %.3f' % (epoch + 1, reg_mse))
                plt.plot(range(len(X_t)), y_t, label='real')
                plt.plot(range(len(X_t)), prediction, label='predict')
                plt.legend()
                plt.show()

                prediction = model.predict(x_test).cpu().detach().numpy()
                reg_loss = get_mae(y_test, prediction)
                reg_mse = get_mse(y_test, prediction)
                print('Epoch: %i; Test MAE loss on source data is: %.3f' % (epoch + 1, reg_loss))
                print('Epoch: %i; Test MSE loss on source data is: %.3f' % (epoch + 1, reg_mse))
                plt.plot(range(len(x_test)), y_test, label='real')
                plt.plot(range(len(x_test)), prediction, label='predict')
                plt.legend()
                plt.show()

        plt.plot(range(len(mae_loss)), mae_loss, label='loss converage')
        plt.show()
        plt.plot(range(len(mse_loss)), mse_loss, label='loss converage')
        plt.show()

        print('------------Domain Adaptation------------')
        disc_losslist, ceod_losslist, loss_list = [], [], []
        mae_loss, mse_loss = [], []
        #         lr = 0.0001
        #         lr1=0.000001
        #         batch_size =64
        #         b1=0.99
        #         b2=0.999
        # #         opt_feat=torch.optim.SGD([{'params': model.feature_extractor.parameters()}], lr=0.0001, momentum=0.8)
        # #         opt_pred=torch.optim.SGD([{'params': model.h_pred.parameters()}], lr=0.0001, momentum=0.8)
        #         opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}],lr=lr,betas=(b1, b2))
        #         opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}],lr=lr,betas=(b1, b2))
        #         opt_disc = torch.optim.Adam([{'params': model.h_disc.parameters()}],lr=lr,betas=(b1, b2))
        #         opt_alpha = torch.optim.Adam([{'params': model.alpha}],lr=lr,betas=(b1, b2))
        #         opt_mu = torch.optim.Adam([{'params': model.mu}],lr=lr,betas=(b1, b2))
        #         opt_beta = torch.optim.Adam([{'params': model.beta}],lr=lr,betas=(b1, b2))
        #         model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha,opt_mu,opt_beta)

        for epoch in range(epochs_adapt):
            running_loss = 0.0
            discloss = 0.0
            ceodloss = 0.0
            model.train()
            loader = multi_data_loader(source_insts, source_labels, batch_size)
            for x_bs, y_bs in loader:
                for j in range(num_domains):
                    x_bs[j] = x_bs[j].reshape(x_bs[j].shape[0], 1, x_bs[j].shape[1])
                    x_bs[j] = torch.tensor(x_bs[j], requires_grad=False).type(torch.FloatTensor).to(device)
                    y_bs[j] = torch.tensor(y_bs[j], requires_grad=False).type(torch.FloatTensor).to(device)
                ridx = np.random.choice(target_insts.shape[0], batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = tinputs.reshape(tinputs.shape[0], 1, tinputs.shape[1])
                x_bt = torch.tensor(tinputs, requires_grad=False).type(torch.FloatTensor).to(device)

                # Train h to minimize source loss

                # stage 1
                for a in range(5):
                    disc_loss1 = model.train_disc(x_bs, x_bt, y_bs, clip=1)
                for b in range(1):
                    loss, source_loss, ceod_loss, disc_loss = model.train_all_feat(x_bs, x_bt, y_bs, clip=1)
                #                     print("extractor is:", model.feature_extractor[5][0].weight)
                #                     print("discriminator is ", model.h_disc[0].weight.data)
                #                     print("Hybrid loss is ", loss)
                # stage 2
                # loss,source_loss,ceod_loss,disc_loss=model.train_all(x_bs, x_bt, y_bs, clip=1)
                # maximize discrepancy by opt_disc
                #                 print("###############finish test########################")
                for c in range(1):
                    loss, source_loss, ceod_loss = model.train_all_pred(x_bs, x_bt, y_bs, clip=1)
                for d in range(5):
                    disc_loss = model.train_all_alpha(x_bs, x_bt, y_bs, clip=1)
                #                     print("predictor is ", model.h_pred.weight.data)
                running_loss += loss.item()
                discloss += disc_loss.item()
                ceodloss += ceod_loss.item()
            loss_list.append(running_loss)
            disc_losslist.append(discloss)
            ceod_losslist.append(ceodloss)
            print('Epoch: %i; MSE loss: %.3f' % (epoch, running_loss))
            print('Epoch: %i; disc loss: %.3f' % (epoch, discloss))
            print('Epoch: %i; ceod loss: %.3f' % (epoch, ceodloss))
            model.eval()
            #             print(X_t)
            prediction = model.predict(X_t).cpu().detach().numpy()
            #             print(prediction)
            stopT = get_mae(y_t, prediction)
            mae_loss.append(stopT)
            print(stopT)
            cur_loss = get_mse(y_t, prediction)
            mse_loss.append(cur_loss)
            print(cur_loss)
            print(model.alpha)
            if (epoch + 1) % 10 == 0:
                print('--------------------------results as follows ---------------------------------')
                model.eval()
                prediction = model.predict(X_t).cpu().detach().numpy()
                #                 source_loss, disc = model.compute_loss(x_bs, X_t, y_s)
                reg_loss = get_mae(y_t, prediction)
                print('Epoch: %i; Test MAE loss on target data is: %.3f' % (epoch, reg_loss))
                plt.plot(range(len(X_t)), y_t, label='real')
                plt.plot(range(len(X_t)), prediction, label='predict')
                plt.legend()
                plt.show()

                prediction = model.predict(x_test).cpu().detach().numpy()
                reg_loss = get_mae(y_test, prediction)
                print('Epoch: %i; Test MAE loss on source data is: %.3f' % (epoch, reg_loss))
                plt.plot(range(len(x_test)), y_test, label='real')
                plt.plot(range(len(x_test)), prediction, label='predict')
                plt.legend()
                plt.show()
            if (epoch + 1) % 10 == 0:
                print(
                    '--------------------------regression discrepency and conditional MMD in 10 epoches as follows---------------------------------')
                plt.plot(range(len(loss_list)), disc_losslist, label='discripency')
                plt.show()
                plt.plot(range(len(loss_list)), ceod_losslist, label='CEOD loss')
                plt.show()
                print(
                    '-------------------------- total running losses in 10 epoches as follows---------------------------------')
                plt.plot(range(len(loss_list)), loss_list, label='Total_loss')
                plt.show()

        plt.plot(range(len(mae_loss)), mae_loss, label='loss converage')
        plt.show()
        plt.plot(range(len(mse_loss)), mse_loss, label='loss converage')
        plt.show()
        model.eval()
        prediction = model.predict(X_t).cpu().detach().numpy()
        stopT = get_mae(y_t, prediction)
        print(stopT)
        print(get_sae(y_t, prediction))
        print(get_nde(y_t, prediction))
        print(model.alpha)
        torch.save(model,"app.pt")