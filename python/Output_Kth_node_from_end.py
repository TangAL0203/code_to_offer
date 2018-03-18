# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if k<=0:
            return None
        if not head:
            return None
        p1 = head
        p2 = head
        for i in range(k-1):
            p2 = p2.next
            if not p2:
                return None
        while(p2.next):
            p1 = p1.next
            p2 = p2.next
        return p1