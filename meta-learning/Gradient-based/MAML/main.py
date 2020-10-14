'''
File: main.py
Project: Reptile
===========
File Created: Wednesday, 14th October 2020 12:51:10 pm
Author: <<LanLing>> (<<lanlingrock@gmail.com>>)
===========
Last Modified: Wednesday, 14th October 2020 1:03:47 pm
Modified By: <<LanLing>> (<<lanlingrock@gmail.com>>)
===========
Description: Reptile 实现 Sine 函数拟合
Copyright <<projectCreationYear>> - 2020 Your Company, <<XDU>>
Ref: https://github.com/gabrielhuang/reptile-pytorch/blob/master/reptile_sine.ipynb
'''

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


# 产生函数
class SineWaveTask:
    def __init__(self):
        # 均匀分布
        self.a = np.random.uniform(0.1, 5.0)
        self.b = np.random.uniform(0, 2*np.pi)
        
    def f(self, x):
        return self.a * np.sin(x + self.b)
        
    def training_set(self, size=10):
        self.train_x = np.random.uniform(-5, 5, size).reshape(size, 1)
        x = self.train_x
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)
    
    def test_set(self, size=5):
        x = np.linspace(-5, 5, size).reshape(size, 1)
        y = self.f(x)
        return torch.Tensor(x), torch.Tensor(y)
    
    def plot(self, filename, dpi=250):
        x, y = self.test_set(size=100)
        plt.plot(x.numpy(), y.numpy())
        plt.savefig(filename, dpi=dpi)

    def plot_model(self, new_model, label, filename):
        x, y_true = self.test_set(size=100)

        y_pred = new_model(x)

        plt.plot(x.data.numpy().flatten(), y_pred.data.numpy().flatten(), label=label)
        plt.legend()
        plt.savefig(filename, dpi=250)


class SineModel(nn.Module):
    
    def __init__(self):
        nn.Module.__init__(self)
        
        self.process = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),            
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
    def forward(self, x):
        return self.process(x)


def do_base_learning(model, wave, lr_inner, n_inner):
    new_model = SineModel()
    new_model.load_state_dict(model.state_dict())  # copy? looks okay
    inner_optimizer = torch.optim.SGD(new_model.parameters(), lr=lr_inner)
    # K steps of gradient descent
    for i in range(n_inner):

        x, y_true = wave.training_set()

        y_pred = new_model(x)

        loss = ((y_pred - y_true)**2).mean()

        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
    return new_model


def do_base_eval(new_model, wave):
    x, y_true = wave.test_set()

    y_pred = new_model(x)
    loss = ((y_pred - y_true)**2).mean()

    return loss
    
        
def reptile_sine(model, iterations, lr_inner=0.01, 
                 lr_outer=0.001, n_inner=3, task_nums=5):
    # 全局优化器
    weights = list(model.parameters())
    optimizer = torch.optim.Adam(weights, lr=lr_outer)

    train_metalosses =[]
    test_metalosses = []
    
    # Sample an epoch by shuffling all training tasks
    for t in range(iterations+1):
        
        new_model = 0
        train_metaloss = 0
        # 多个任务
        for _ in range(task_nums):
            # Sample task
            wave = random.sample(SINE_TRAIN, 1)[0]
            # Take k gradient steps
            new_model = do_base_learning(model, wave, lr_inner, n_inner)
            # Eval
            train_metaloss += do_base_eval(new_model, wave)
        
        # 根据累计误差，计算梯度值
        meta_grads = torch.autograd.grad(train_metaloss, weights, allow_unused=True)
        # print('here')
        
        # 更新权重的梯度值
        for w, g in zip(weights, meta_grads):
            w.grad = g
        # Update meta-parameters
        optimizer.step()
        optimizer.zero_grad()
    
        ############# Validation
        wave = random.sample(SINE_TEST, 1)[0]
        new_model = do_base_learning(model, wave, lr_inner, n_inner)
        test_metaloss = do_base_eval(new_model, wave)
        
        ############# Log
        train_metalosses.append(train_metaloss)
        test_metalosses.append(test_metaloss)
        
        if t % 1000 == 0:
            print('Iteration {:<5d}'.format(t), 'AvgTrainML ', round(np.mean(train_metaloss.item())/task_nums, 2), 
            'AvgTestML ', round(np.mean(test_metaloss.item()), 2))


TRAIN_SIZE = 10000
TEST_SIZE = 1000
SINE_TRAIN = [SineWaveTask() for _ in range(TRAIN_SIZE)]
SINE_TEST = [SineWaveTask() for _ in range(TEST_SIZE)]


# 创建模型
model = SineModel()

# 开始训练
reptile_sine(model, iterations=5000)

wave = SineWaveTask()
# 画一下原始函数图像
wave.plot(filename='sine.png')


for n_inner_ in range(4):
    new_model = do_base_learning(model, wave, 
                                 lr_inner=0.01, n_inner=n_inner_)
    wave.plot_model(new_model, label=str(n_inner_)+' gradient steps', filename=str(n_inner_)+'.png')
