# -*- coding:utf-8 -*-

# 基数排序

import math
# 这里基数为10，数为十进制
def radix_sort(lists, radix=10):
    k = int(math.ceil(math.log(max(lists), radix))) # 数组元素可以由多少位来表示，用d元组来表示
    bucket = [[] for i in range(radix)] # 创建radix(基数)个队列
    for i in range(1, k+1):
        for j in lists:
            bucket[j/(radix**(i-1)) % (radix)].append(j) # 根据数组元素在某个基上的值，将其添加到该基对应的队列中
        del lists[:] # 将lists元素清空
        for z in bucket:
            lists += z # 更新lists元素，按照第i基增量顺序来进行首尾相加
            del z[:] # 清空队列中的元素，bucket重新变为[[]]
    return lists

lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print radix_sort(lists)