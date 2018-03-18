#-*-coding:utf-8-*-
#变态跳台阶问题，可以归纳出规律，1 2 4 8 16 ...
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number<=1:
            return 1
        result = 1
        for i in range(1,number):
            result = result*2
        return result