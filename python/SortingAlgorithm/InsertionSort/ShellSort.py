# -*- coding:utf-8 -*-
def shell_sort(lists):
    # 希尔排序
    count = len(lists)
    step = 2
    group = count / step
    while group > 0:
        # i equals first group positions
        for i in range(0, group):
            j = i + group # 间隔为group为同一组元素，递增
            while j < count:
                k = j - group # 记录j前面的元素
                key = lists[j]
                while k >= 0:
                    if lists[k] > key:
                        lists[k + group] = lists[k]
                        lists[k] = key
                    k -= group
                j += group
        group /= step
    return lists

#测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print shell_sort(lists)