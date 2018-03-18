# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        pHead = None
        if pHead1.val<pHead2.val:
            pHead = pHead1
            pHead.next = Merge(pHead1.next, pHead2)
        else:
            pHead = pHead2
            pHead.next = Merge(pHead1, pHead2.next)

        return pHead

## 复杂的解法
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 is None:
            return pHead2
        if pHead2 is None:
            return pHead1
        
        if pHead1.val<=pHead2.val:
            first_Head = pHead1 # 待返回的链表头部
            temp = pHead1 # 当前最小值
            another_temp = pHead2 # 另一个链表最小值
            while temp.next:
                if temp.next.val>=another_temp.val:
                    temp3 = temp.next # 用于交换的临时变量
                    temp.next = another_temp
                    temp = another_temp
                    another_temp = temp3
                else:
                    temp = temp.next
            if another_temp:
                temp.next = another_temp
            return first_Head
        else:
            first_Head = pHead2 # 待返回的链表头部
            temp = pHead2 # 当前最小值
            another_temp = pHead1 # 另一个链表最小值
            while temp.next:
                if temp.next.val>=another_temp.val:
                    temp3 = temp.next
                    temp.next = another_temp
                    temp = another_temp
                    another_temp = temp3
                else:
                    temp = temp.next
            if another_temp:
                temp.next = another_temp
            return first_Head