import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck


class Compare(nn.Module):
    """
    repnet => feature concat => layer4 & layer5 & avg pooling => fc => sigmoid
    """
    def __init__(self, n_way, k_shot):
        super(Compare, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot

        self.repnet = repnet_deep(False)
        # 返回特征提取的维度
        repnet_sz = self.repnet(Variable(torch.rand(2, 3, 224, 224))).size()
        # 一组训练集或测试集中包含的样本数量
        self.c = repnet_sz[1]
        # 猜测等于2
        # print('self.c is {}'.format(self.c))
        # 图像高度
        self.d = repnet_sz[2]
        # layer4 的输入
        self.inplanes = 2 * self.c
        # 图像高度和宽度是否相等
        assert repnet_sz[2] == repnet_sz[3]
        # print('repnet sz:', repnet_sz)

        # 拼接之后的操作
        self.layer4 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(256 , 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    # 训练：开始比较
    def forward(self, support_x, support_y, query_x, query_y, rank, train=True):
        """

        """
        # [多少组数组，每个类的样本数量，3个通道，图片高度，图片宽度]
        batchsz, setsz, c_, h, w = support_x.size()
        # 每组数据中 query 样本的数量
        querysz = query_x.size(1)
        # c 是 batchsize，d 是图片高度
        c, d = self.c, self.d

        # view 重构张量的维度
        # batchsz * setsz 是输入的 batchsize 大小
        # [b, setsz, c_, h, w] => [b*setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
        support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
        # [b, querysz, c_, h, w] => [b*querysz, c_, h, w] => [b*querysz, c, d, d] => [b, querysz, c, d, d]
        query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

        # 按照图片的三个通道，将查询集和测试集拼接起来
        # squeeze：删除维度是 1 的数据
        # unsqueeze：在指定位置插入一维数据
        # expand 在指定维度进行扩展
        # 第一个维度，扩展查询集这么多个。[b, setsz, c, d, d] => [b, 1, setsz, c, d, d] => [b, querysz, setsz, c, d, d]
        support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
        # 第二个维度，扩展支持集这么多个。[b, querysz, c, d, d] => [b, querysz, 1, c, d, d] => [b, querysz, setsz, c, d, d]
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)
        # cat: [b, querysz, setsz, c, d, d] => [b, querysz, setsz, 2c, d, d]
        comb = torch.cat([support_xf, query_xf], dim=3)

        # print('comb size is {}'.format(comb.size()))

        # print('c = {}'.format(c))
        comb = self.layer5(self.layer4(comb.view(batchsz * querysz * setsz, 2 * c, d, d)))
        # print('layer5 sz:', comb.size())
        comb = F.avg_pool2d(comb, 3)
        # print('avg sz:', comb.size())
        # push to Linear layer
        # [b * querysz * setsz, 256] => [b * querysz * setsz, 1] => [b, querysz, setsz, 1] => [b, querysz, setsz]
        score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3)

        # build its label
        # [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
        support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
        # [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
        query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
        # 标签相等，表示查询集属于这个类。和score进行对比
        # eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
        label = torch.eq(support_yf, query_yf).float()

        # score: [b, querysz, setsz]
        # label: [b, querysz, setsz]
        if train:
            loss = torch.pow(label - score, 2).sum() / batchsz
            return loss

        else:
            # [b, querysz, setsz]
            rn_score_np = score.cpu().data.numpy()
            pred = []
            # [b, setsz]
            support_y_np = support_y.cpu().data.numpy()
            for i, batch in enumerate(rn_score_np):
                for _, query in enumerate(batch):
                    # query: [setsz]
                    sim = []  # [n_way]
                    for way in range(self.n_way):
                        sim.append(np.sum(query[way * self.k_shot: (way + 1) * self.k_shot]))
                    idx = np.array(sim).argmax()
                    pred.append(support_y_np[i, idx * self.k_shot])
            # pred: [b, querysz]
            pred = Variable(torch.from_numpy(np.array(pred).reshape((batchsz, querysz)))).cuda().to(rank)

            correct = torch.eq(pred, query_y).sum()
            return pred, correct
