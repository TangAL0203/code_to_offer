# -*- coding:utf-8 -*-
def insert_sort(lists):
    # 插入排序
    # 将列表分为[有序+无序部分]，每次插入的时候，从无序部分中取第一个元素插入到合适的位置
    # 长度为n的列表，需要进行n-1次插入，每次插入的位置不是固定的
    count = len(lists)
    if len(lists)<=1:
        return lists
    for i in range(1,len(lists)):
        key = lists[i]
        j = i-1
        while j>=0:
            if lists[j]>key:
                lists[j+1] = lists[j]
                lists[j] = key
            key = lists[j]
            j-=1

    return lists

#测试
lists=[10,23,1,53,654,54,16,646,65,3155,546,31]
print insert_sort(lists)
