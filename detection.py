import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# box1: N*4；box2: M*4。
# 计算box1集合和box2集合的IOU值。要求不要使用for循环。

## 单次for循环的例子

import numpy as np

box1 = np.array([[1, 1, 3, 3], [4, 4, 8, 8]])
box2 = box1.copy()

N, M = box1.shape[0], box1.shape[0]

ans = np.zeros((N, M))
eps = 1e-6
area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

for i, box in enumerate(box1):
    x1 = np.maximum(box[0], box2[:, 0])
    y1 = np.maximum(box[1], box2[:, 1])
    x2 = np.minimum(box[2], box2[:, 2])
    y2 = np.minimum(box[3], box2[:, 3])

    w, h = np.maximum(0, x2 - x1 + 1), np.maximum(0, y2 - y1 + 1)
    inter = w * h

    over = inter / (area1[i] + area2 - inter + eps)

    ans[i, :] = over


## 广播，不用for循环例子
def calculate_iou_matrix(boxes1, boxes2):
    # 转换为NumPy数组
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    # 计算左上角和右下角坐标的交集
    intersection_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    intersection_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    intersection_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    intersection_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    # 计算交集面积，并设置非法交集为0
    intersection_area = np.maximum(intersection_x2 - intersection_x1, 0) * np.maximum(
        intersection_y2 - intersection_y1, 0)

    # 计算两个矩形的并集面积
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 计算IOU矩阵
    iou = intersection_area / (boxes1_area[:, None] + boxes2_area - intersection_area)

    return iou


# 示例使用
boxes1 = [[0, 0, 1, 1], [2, 2, 4, 4]]
boxes2 = [[1, 1, 2, 2], [3, 3, 5, 5]]
iou_matrix = calculate_iou_matrix(boxes1, boxes2)
print(iou_matrix)
