# -*- coding:utf-8 -*-
#quick sort
def quick_sort(lists, left, right):
    # 快速排序
    if left >= right:
        return lists
    key = lists[left]
    low = left
    high = right
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]
    lists[right] = key # 将key值归位
    quick_sort(lists, low, left - 1)
    quick_sort(lists, left + 1, high)
    return lists

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print quick_sort(lists,0,len(lists)-1)