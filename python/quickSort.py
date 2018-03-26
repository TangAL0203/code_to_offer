# -*- coding:utf-8 -*-
#quick sort
def quickSort(L, low, high):
    i = low 
    j = high
    if i >= j:
        return L
    key = L[i]  # key值是不变的
    while i < j:
        while i < j and L[j] >= key:
            j = j-1
        # 当L[j]<key时，将该值与key替换   
        temp = L[i]                                                          
        L[i] = L[j] # i处的值变为更小的L[j]了
        L[j] = temp
        while i < j and L[i] <= key:
            i = i+1
        temp = L[j]
        L[j] = L[i]
        L[i] = temp
    L[i] = key 
    quickSort(L, low, i-1)
    quickSort(L, j+1, high)
    return L