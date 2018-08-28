# -*- coding:utf-8 -*-
# 折半插入排序
def BinaryInsertSort(array):
    if len(array)<=1:
        return array
    count = len(array)
    for i in range(1,count):
        low = 0
        high = i-1 # low和high表示已经排好序的区间
        value = array[i]
        # 将待插入的元素和之前拍好序的元素做比较，二分缩小区间，找到合适的插入位置，插入位置是low
        while low<=high:
            mid = (low+high)/2
            if value>array[mid]:
                low = mid+1
            else:
                high = mid-1
        # 
        # 先移动后面的，再移动前面的
        for j in range(i-1,low-1,-1):
            array[j+1] = array[j]
        array[low] = value
    return array

#测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print BinaryInsertSort(lists)
