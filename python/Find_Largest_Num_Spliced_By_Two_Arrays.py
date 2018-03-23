#-*- coding:utf-8 -*-
def get_max(L):
    if len(L)==0:
        return False
    str_L = [str(i) for i in L]
    quickSort(str_L, 0, len(str_L)-1)
    result = ''
    # from end to start, return max
    # from start to end, return min
    for i in range(len(str_L)-1,-1,-1):
        result += str_L[i]
    return result 

def quickSort(L, low, high):
    i = low 
    j = high
    if i >= j:
        return L
    key = L[i]  
    while i < j:
        linked_num1 = key + L[j]
        linked_num2 = L[j] + key
        while i < j and linked_num2  >= linked_num1 :
            j = j-1
        temp = L[i]                                                          
        L[i] = L[j] 
        L[j] = temp
        linked_num1 = key + L[i]
        linked_num2 = L[i] + key
        while i < j and linked_num2  <= linked_num1 :
            i = i+1
        temp = L[j]
        L[j] = L[i]
        L[i] = temp
    L[i] = key 
    quickSort(L, low, i-1)
    quickSort(L, j+1, high)
    return L
    