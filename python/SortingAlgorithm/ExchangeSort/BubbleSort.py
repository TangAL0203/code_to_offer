#-*-coding:utf-8-*-
def BubbleSort(array):
    count = len(array)
    if count<=1:
        return array
    # i控制比较的范围和次数，冒泡需要比较n-1趟
    for i in range(count-2,-1,-1):
        for j in range(0,i+1):
            if array[j]>array[j+1]:
                temp = array[j+1]
                array[j+1] = array[j]
                array[j] = temp
    return array

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print BubbleSort(lists)
