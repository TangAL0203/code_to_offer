# -*- coding:utf-8 -*-

# 对两个有序表进行归并操作
# left为左有序表
# right为右有序表
def merge(left, right):
    i, j = 0, 0 # i和j指向列表的start位置
    result = [] # 创建一个辅助列表
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:] # 把剩下的元素添加到result末尾即可
    result += right[j:]
    return result
 
def merge_sort(lists):
    # 归并排序
    if len(lists) <= 1:
        return lists
    num = len(lists) / 2
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    return merge(left, right)

# 测试
lists=[1,10,2,8,23,1,53,654,54,16,646,65,3155,546,31]
print merge_sort(lists)