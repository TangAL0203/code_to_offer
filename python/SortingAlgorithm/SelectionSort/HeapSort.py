# -*- coding:utf-8 -*-

def adjust_heap(lists, i, size):
    # lists 需构建堆的数组
    # i 当前根结点坐标
    # size 数组的长度
    lchild = 2 * i + 1
    rchild = 2 * i + 2
    max = i
    if i < size / 2:
        # 找根结点与左右子结点的最大值对应的坐标
        if lchild < size and lists[lchild] > lists[max]:
            max = lchild
        if rchild < size and lists[rchild] > lists[max]:
            max = rchild
        if max != i:
            lists[max], lists[i] = lists[i], lists[max] # 交换，满足当前根结点值大于孩子结点
            adjust_heap(lists, max, size) # 可能会破坏下一级堆，需递归地构造下一级的堆

def build_heap(lists, size):
    for i in range(0, (size/2))[::-1]:
        adjust_heap(lists, i, size)

def heap_sort(lists):
    size = len(lists)
    build_heap(lists, size) # 构建初始堆，堆中元素个数为lists元素个数
    for i in range(0, size)[::-1]:
        lists[0], lists[i] = lists[i], lists[0] # 堆顶元素与堆尾互换
        adjust_heap(lists, 0, i) # 调整堆，堆的元素减一
    return lists

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print heap_sort(lists)