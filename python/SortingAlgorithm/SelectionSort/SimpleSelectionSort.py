# -*- coding:utf-8 -*-
# 简单选择排序
# 思路：从前往后遍历（升序），每次当前的list中找到最小的元素（实际是找到相应的位置），然后将最小元素和遍历到的位置进行交换
def SimpleSelectSort(lists):
    if len(lists)<=1:
        return lists
    for i in range(0,len(lists)-1):
        minXPos = i
        for j in range(i+1,len(lists)):
            if lists[j]<lists[minXPos]:
                minXPos = j
        # swap
        temp = lists[i]
        lists[i] = lists[minXPos]
        lists[minXPos] = temp
    return lists

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print SimpleSelectSort(lists)
