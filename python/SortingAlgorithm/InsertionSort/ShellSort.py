# -*- coding:utf-8 -*-
def shell_sort(lists):
    # 希尔排序
    count = len(lists)
    step = 2
    group = count / step
    while group > 0:
        # i equals first group positions
        # 间隔为group为同一组元素，
        for i in range(0, group):
            # 对某一组元素做插入排序
            j = i + group # 该组中需要插入的第一个元素
            while j < count:
                k = j - group # 记录j前面的元素
                key = lists[j]
                while k >= 0:
                    if lists[k] > key:
                        lists[k + group] = lists[k]
                        lists[k] = key
                    k -= group
                j += group # 更新需要插入的元素
        group /= step
    return lists

#测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print shell_sort(lists)
