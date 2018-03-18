#-*-coding:utf-8-*-
#简单跳台阶问题，可以归纳出规律，1 2 3 5 8 ...
class Solution:
    def jumpFloor(self, number):
        # write code here
        a0 = 1
        a1 = 1
        if number<=1:
            return 1
        for i in range(1,number):
            a2 = a1+a0
            a1,a0 = a2,a1
        return a1