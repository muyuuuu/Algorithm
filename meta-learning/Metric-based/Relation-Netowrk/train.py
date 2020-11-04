import torch
import os
import time
import numpy as np
from torch import optim
from torchsummary import summary
from torch.autograd import Variable
from compare import Compare
from miniImagenet import MiniImagenet
from torch.utils.data import DataLoader


if __name__ == '__main__':

    '''
    超参数定义
    '''
    # 5 个类
    n_way = 5
    # 1 个shot
    k_shot = 1
    # 每个类的查询样本
    k_query = 1
    # 选择 batchsz 组 [训练集，测试集]
    batchsz = 30
    # 输出训练过程中最好的准确性
    best_accuracy = 0

    # 数据并行化
    print('parameters setting over.')
    # print('To run on single GPU, change device_ids=[0] and downsize batch size! \nmkdir ckpt if not exists!')
    
    # GPU 单卡
    # net = torch.nn.DataParallel(Compare(n_way, k_shot), device_ids=[0]).cuda()
    
    # GPU 单机多卡
    net = torch.nn.DataParallel(Compare(n_way, k_shot)).cuda()

    # CPU 版本
    # net = Compare(n_way, k_shot)

    # 打印模型
    print(net)

    # 模型参数要保留的文件名
    mdl_file = '../ckpt/compare%d%d.mdl' % (n_way, k_shot)

    # 如果有之前的经验，则加载，否则直接运行
    if os.path.exists(mdl_file):
        print('load checkpoint ...', mdl_file)
        net.load_state_dict(torch.load(mdl_file))

    # 选择需要梯度更新的参数
    print("The numbers of parameter model: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # 创建优化器
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 开始训练
    print('Train is beginning...')
    since = time.time()
    for epoch in range(200):
        print('Begin to load data...')
        # 加载训练数据集 batchsz 组支持集和查询集
        mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
                            batchsz=10000, resize=224)
        # pin_memory 快速的将数据转化为 GPU 可以处理的数据
        # num_workers 读取数据子线程的数量
        db = DataLoader(mini, batchsz, shuffle=True,
                        num_workers=8, pin_memory=True)
        # 加载测试数据集
        mini_val = MiniImagenet('../mini-imagenet/', mode='val', n_way=n_way, k_shot=k_shot, k_query=k_query,
                                batchsz=200, resize=224)
        db_val = DataLoader(mini_val, batchsz, shuffle=True,
                            num_workers=2, pin_memory=True)

        print('Support and Query set is entranced...')
        # 训练阶段。遍历训练数据集中的每一个 batch
        for step, batch in enumerate(db):
            # 支持集合与查询集合
            support_x = Variable(batch[0]).cuda()
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2]).cuda()
            query_y = Variable(batch[3]).cuda()
            # 开始训练
            net.train()
            # 计算 loss
            # print('computing loss....')
            loss = net(support_x, support_y, query_x, query_y)
            # Multi-GPU support
            loss = loss.mean()

            # 清空优化器之前的梯度
            optimizer.zero_grad()
            # 反向传播
            # print('backwarding ...')
            loss.backward()
            optimizer.step()

            # 开始验证
            total_val_loss = 0
            if step % 200 == 0:
                total_correct = 0
                total_num = 0
                # 在测试集中
                for j, batch_test in enumerate(db_val):
                    # 选出支持集和查询集
                    support_x = Variable(batch_test[0]).cuda()
                    support_y = Variable(batch_test[1]).cuda()
                    query_x = Variable(batch_test[2]).cuda()
                    query_y = Variable(batch_test[3]).cuda()
                    # 评价
                    net.eval()
                    pred, correct = net(
                        support_x, support_y, query_x, query_y, False)
                    correct = correct.sum()  # multi-gpu support
                    # 累计的准确率
                    total_correct += correct.item()
                    # query_y.size(0) 表示这么多组的查询集
                    # query_y.size(0) 每个查询集中样本的数量
                    # total_num 就是一共多少测试样本
                    total_num += query_y.size(0) * query_y.size(1)

                # 计算准确率
                accuracy = total_correct / total_num
                # 保留最好的准确率
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(net.state_dict(), mdl_file)
                    # print('saved to checkpoint:', mdl_file)

                print('<<<<>>>>accuracy:', accuracy,
                      'best accuracy:', best_accuracy)
                with open('../output/accuracy.txt', 'a') as f:
                    f.write('accuracy: {}\n'.format(accuracy))

            # 每 15 轮打印一次 loss 函数
            if step % 15 == 0 and step != 0:
                print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' % (
                    n_way, k_shot, batchsz, epoch, step, loss.cpu().item()))
                with open('../output/loss.txt', 'a') as f:
                    f.write('loss: {}\n'.format(loss.cpu().item()))

    summary(net, (2, 3, 234, 234))
    end = time.time()
    print("cost time is {}".format(end - since))
