import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv


class MiniImagenet(Dataset):
    """
    加载数据的类
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """
        :param root: mini-imagenet 的存储路径
        :param mode: 训练还是测试
        :param batchsz: 指定每次训练有多少组数据
        :param n_way: 类的数量
        :param k_shot: 每个类中样本的数量
        :param k_query: 每个类查询样本的数量
        :param resize: 数据大小重塑
        :param startidx: 从 startidx 开始索引数据
        """

        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        # 训练时，每个集合样本的数量
        self.setsz = self.n_way * self.k_shot
        # 测试时，每个集合的样本数量
        self.querysz = self.n_way * self.k_query
        self.resize = resize
        self.startidx = startidx
        print('%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' %
              (mode, batchsz, n_way, k_shot, k_query, resize))

        # 对图像完成转换操作
        # x 是文件路径
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize(
                                                 (self.resize, self.resize)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])

        # 图片路径
        self.path = os.path.join(root, 'images')
        # 加载 CSV 文件，返回字典类型
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            # 同一类标签的数据在列表中的索引一样
            self.data.append(v)
            #  将标签转化为类别，类别编号是 i
            self.img2label[k] = i + self.startidx

        # 计算有多少类别的数据
        self.cls_num = len(self.data)

        # 创建数据
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        返回字典类型，字典的键是标签，值是这个标签对应的所有文件名
        输入参数：mini-imagenet带有的csv文件
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            # 跳过第一行
            next(csvreader, None)
            # 字典按照标签添加文件名
            for _, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # 标签是 key，文件名是值
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        功能描述：创建 batchsz 组支持集和测试集
        """
        # 支持集
        self.support_x_batch = [] 
        # 查询集
        self.query_x_batch = []

        # 生成 batchsz 组 [支持集，测试集]
        for _ in range(batchsz):
            # 选择 n_way 个类的数据
            selected_cls = np.random.choice(
                self.cls_num, self.n_way, False)  
            
            support_x = []
            query_x = []

            # 在选择的类中
            for cls in selected_cls:
                # 为每一个类选择 shot 大小的数据 和 query 大小的数据
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                # 训练数据的索引
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])
                # 测试数据的索引 
                indexDtest = np.array(selected_imgs_idx[self.k_shot:]) 
                
                # 添加支持集
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())
                # 添加查询集
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # 全部支持集
            # 可以理解为一个二维数组，每一行里面是一组支持集，元素为文件名
            self.support_x_batch.append(support_x)
            # 全部查询集
            self.query_x_batch.append(query_x)

    # MiniImagenet 类切片访问
    def __getitem__(self, index):
        """
        功能描述：从batchsz组数据里面，返回索引对应的数据（一维）
        输入参数：
            index：索引
        """
        # [每个类的样本数量，3个通道，图片尺寸]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # 支持集不知道属于哪个类，暂时用 0 填充
        support_y = np.zeros((self.setsz), dtype=np.int)
        # 查询集同理
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        query_y = np.zeros((self.querysz), dtype=np.int)
        
        # 遍历选择的支持集中的每一个元素，记录其路径
        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]

        # 记录选择的支持集中每一个元素，第九个元素之后为标签，按照 img2label 生成标签
        support_y = np.array([self.img2label[item[:9]]
             for sublist in self.support_x_batch[index] for item in sublist])

        # 查询集同理
        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist])

        # 对文件进行转化
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        
        # 返回需要的切片数据
        return support_x, torch.LongTensor(torch.from_numpy(support_y)), \
            query_x, torch.LongTensor(torch.from_numpy(query_y))

    def __len__(self):
        return self.batchsz
