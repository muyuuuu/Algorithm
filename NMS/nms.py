import numpy as np


def NMS(dets, thresh=0.4):
    '''
    NMS 算法
    输入：
        dets: x1, y1, x2, y2, score
    输出：
        保留的目标框的索引
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个候选框的面积
    areas = (x2 - x1) * (y2 - y1)

    # scores [0.6 0.4 0.5 0.7 1. ]
    # 排序后 [1. 0.7 0.6 0.5 0.4]
    # 排序后的元素对应原来数组的索引 [4 3 0 2 1]
    order = scores.argsort()[::-1]

    temp = []
    while order.size > 0:
        # 当前概率最大的盒子的索引
        i = order[0]
        temp.append(i)
        # 除被保留的盒子外，其余盒子的 (xx1, yy1, xx2, yy2) 计算
        # 交集位于 (x1, y1, x2, y2) 之间

        # x1 计算，如果比 x1 大，不用管。比 x1 小，设置为 x1 的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        # x2 计算，如果比 x2 小，不用管。比 x2 大，设置为 x2 的坐标
        xx2 = np.minimum(x2[i], x2[order[1:]])
        # 如果比 y1 大，不用管。比 y1 小，设置为 y1 的坐标
        yy1 = np.maximum(y1[i], y1[order[1:]])
        # 如果比 y2 小，不用管。比 y2 大，设置为 y2 的坐标
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        #计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # 这些盒子继续 NMS
        # +1是因为，有一个盒子不会被删除，其余盒子都是往后顺移了一个
        order = order[inds + 1]
    return temp


if __name__ == "__main__":
    dets = np.array([[10, 10, 30, 30, 0.6], [30, 10, 50, 30, 0.4],
                     [10, 30, 30, 50, 0.5], [30, 30, 50, 50, 0.7],
                     [20, 20, 40, 40, 1]])
    # 设置阈值
    thresh = 0.1
    keep_dets = NMS(dets, thresh)
    # 打印留下的框的索引
    print(keep_dets)
    # 打印留下的框的信息
    print(dets[keep_dets])