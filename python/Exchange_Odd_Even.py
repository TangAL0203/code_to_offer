# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        if array is None:
            return array
        if len(array)<=1:
            return array
        p1 = 0
        p2 = len(array)-1
        while(abs(p2-p1)>=1):
            if array[p1]%2==1:
                p1+=1
            else:
                while(array[p2]%2==0):
                    p2-=1
                temp = array[p1]
                array[p1] = array[p2]
                array[p2] = temp
        return array