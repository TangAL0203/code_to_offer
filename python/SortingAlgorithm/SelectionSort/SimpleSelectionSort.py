# -*- coding:utf-8 -*-

def SimpleSelectSort(array):
    count = len(array)
    if count<=1:
        return array
    for i in range(count):
        minpos = i
        # get pos of min value
        for j in range(i,count):
            if array[j]<array[minpos]:
                minpos =j
        # switch min value and pos i
        temp = array[i]
        array[i] = array[minpos]
        array[minpos] = temp
        
    return array

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print SimpleSelectSort(lists)
