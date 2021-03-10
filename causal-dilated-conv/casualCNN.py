import torch
import numpy


class Chomp1d(torch.nn.Module):
    """
    删除时序序列的最后 s 个元素
    输入的是 (B, C, L)
    输出的是 (B, C, L-s)
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # print(x.size(), self.chomp_size, x[:, :, :-self.chomp_size].size())
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    三维向量去除第三维，返回二维
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    单个卷积块
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    输入 (B, C, L)
    输出 (B, C, L)

    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # 计算填充，使所应用的卷积为因果关系
        padding = (kernel_size - 1) * dilation

        # 第一个卷积块
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # 截断使卷积成为因果关系
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # 第二个卷积块
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # 卷积网络
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # 残差连接
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # 最终的激活函数
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        print(out_causal.size())
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    卷积块组成卷积网络

    输入是 (B, C, L)
    输出也是 (B, C, L)

    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = [] 
        dilation_size = 1

        # 三层 CNN 
        for i in range(depth):
            # 每一层的输入维度和输出维度
            # 中间几层的输入和输出通道数一致
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            # 步长扩大
            dilation_size *= 2

        # 最后一层
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    因果扩张CNN将输出压缩到固定尺寸，然后自适应 maxpool 后连接全连接输出。

    输入维度是(B, C, L)，输出维度是(B, C)

    @param in_channels  输入数据的通道数
    @param channels     CNN操纵的通道数，也可以理解为隐藏层输出的通道数
    @param depth        CNN的深度
    @param reduced_size CNN输出减少到的固定尺寸
    @param out_channels 输出通道数
    @param kernel_size  卷积核的尺寸大小
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        self.causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        # 池化最后一个维度到尺寸 1 其实也可以靠卷积到尺寸 1
        self.reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        # 取消第三个维度
        self.squeeze = SqueezeChannels()
        # 线性层，输出指定维度的特征
        self.linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            self.causal_cnn, self.reduce_size, self.squeeze, self.linear
        )

    def forward(self, x):
        x = self.causal_cnn(x)
        x = self.reduce_size(x)
        x = self.squeeze(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":

    # bacth，in_channel，length
    x = torch.rand((32, 4, 12))
    # in_channel，hidden_channel, layer, out_channel, output_dim, kernel_size
    encoderNet = CausalCNNEncoder(4, 40, 3, 160, 320, 3)
    y = encoderNet(x)