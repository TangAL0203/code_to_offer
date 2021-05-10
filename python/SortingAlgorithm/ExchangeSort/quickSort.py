# -*- coding:utf-8 -*-
#quick sort
def quick_sort(lists, left, right):
    # 快速排序
    if left >= right:
        return
    key = lists[left]
    low = left
    high = right
    # 一趟快排过程
    while left < right:
        while left < right and lists[right] >= key:
            right -= 1
        lists[left] = lists[right]  # 将right的值传到左边区间, 这时key值丢失(后面有归位操作)
        while left < right and lists[left] <= key:
            left += 1
        lists[right] = lists[left]  # 将left值传到右边区间, 虽然替换了right的值, 但是之前right的值已经传递到左边区间了, 所以right值不会丢失
    lists[right] = key # 将key值归位，此时left==right
    quick_sort(lists, low, left - 1)
    quick_sort(lists, left + 1, high)
    return lists

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print quick_sort(lists,0,len(lists)-1)
