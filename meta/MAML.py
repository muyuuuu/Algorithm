import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from tasks import Sine_Task, Sine_Task_Distribution
import matplotlib.pyplot as plt


# 实现 MAMLModel
class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1, 10)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(10, 1)),
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        return x


class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        self.model = model # MAML model
        self.tasks = tasks #  Sine_Task_Distributions class
        self.weights = list(model.parameters()) # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
        
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K   #  K 是采样数量
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        self.tasks_per_meta_batch = tasks_per_meta_batch 
        
        # metrics
        self.plot_every = 10
        self.print_every = 50
        self.meta_losses = []
    
    def inner_loop(self, task, temp_weights):
        # reset inner model to current maml weights
        # perform training on data sampled from task
        X, y = task.sample_data(self.K)
        # 模型更新参数的次数
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K
            
            # 内部更新梯度 theta
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        
        # 重新采样并计算误差
        X, y = task.sample_data(self.K)
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K
        
        return loss, temp_weights
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0
            temp_weights = [w.clone() for w in self.weights]
            # tasks_per_meta_batch 表示任务数量，这么多任务的损失累加起来
            for i in range(self.tasks_per_meta_batch):
                #  返回 Sine_task 的类
                task = self.tasks.sample_task()
                # 内部循环
                loss, temp_weights = self.inner_loop(task, temp_weights)
                meta_loss += loss
            
            # 根据累计误差，计算梯度值
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # 更新权重的梯度值
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
            # 计算平均误差
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            
            # 50轮打印一次，累计了 plot_every 这么多次，所以除以
            # if iteration % self.print_every == 0:
                # print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            
            # plot_every 这么多次记录一下，最后清 0
            # if iteration % self.plot_every == 0:
                # self.meta_losses.append(epoch_loss / self.plot_every)
                # epoch_loss = 0
            
            print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss))
            epoch_loss = 0

# 创建任务
tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5)
# 
maml = MAML(MAMLModel(), tasks, inner_lr=0.01, meta_lr=0.001)

maml.main_loop(num_iterations=10000)
plt.plot(maml.meta_losses)
plt.savefig("loss.png", dpi=250)


# https://github.com/vmikulik/maml-pytorch/blob/master/MAML-Sines.ipynb