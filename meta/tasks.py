import torch
import torch.nn as nn
import numpy as np


class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, a, b, xmin, xmax):
        self.a = a
        self.b = b
        self.xmin = xmin
        self.xmax = xmax
        
    # 返回波形函数
    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        
        return self.a * np.sin(self.b + x)
        
    # 采样，在 [x_min, x_max] 之间，采样数量为 size
    def sample_data(self, size=1):
        """
        Sample data from this task.
        
        returns: 
            x: the feature vector of length size
            y: the target vector of length size
        """
        
        x = np.random.uniform(self.xmin, self.xmax, size)
        y = self.true_function(x)
        
        # reshape  (size, 1)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)
        
        return x, y


class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """
    
    def __init__(self, a_min, a_max, b_min, b_max, x_min, x_max):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.x_min = x_min
        self.x_max = x_max
        
    def sample_task(self):
        """
        Sample from the task distribution.
        
        returns:
            Sine_Task object
        """
        a = np.random.uniform(self.a_min, self.a_max)
        b = np.random.uniform(self.b_min, self.b_max)
        return Sine_Task(a, b, self.x_min, self.x_max)
