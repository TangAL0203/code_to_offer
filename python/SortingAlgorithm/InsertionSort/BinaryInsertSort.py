# -*- coding:utf-8 -*-
def BinaryInsertSort(array):
    if len(array)<=1:
        return array
    count = len(array)
    for i in range(1,count):
        low = 0
        high = i-1
        value = array[i]
        # find correct inserted position
        while low<=high:
            mid = (low+high)/2
            if value>array[mid]:
                low = mid+1
            else:
                high = mid-1
        # move
        for j in range(i-1,low-1,-1):
            array[j+1] = array[j]
        array[low] = value
    return array

#测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print BinaryInsertSort(lists)
