#-*-coding:utf-8-*-
def BinarySearch(array, t):
    low = 0
    high = len(array)-1
    while low<high:
        mid = (low+high)/2
        if array[mid]<t:
            low = mid+1
        elif array[mid]>t:
            high = mid-1
        else:
            return array[mid]
    return -1