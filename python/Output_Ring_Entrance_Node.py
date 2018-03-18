# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        if not pHead:
            return None
        start = pHead
        p1 = pHead
        p2 = pHead.next
        if not p2:
            return None
        while(p1!= p2 and p2):
            p2 = p2.next.next
            p1 = p1.next
        if not p2:
            return None
        len_ring = 0
        ring_node = p1
        ring_node = ring_node.next
        if ring_node:
            len_ring=2
        while(ring_node!=p1):
            ring_node = ring_node.next
            len_ring+=1
        p1 = pHead
        p2 = pHead
        for i in range(1,len_ring):
            p2 = p2.next
        while(p2!=p1):
            p2 = p2.next
            p1 = p1.next
        return p1