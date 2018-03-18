# -*- coding:utf-8 -*-
# 迭代32次(负数的右移会保持最高位为1不变，因此选择左移)
# 用Python实现有问题，推荐用C和C++实现
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        flag = 1
        while(n):
            if(n&flag):
                count+=1
            flag = flag<<1

# 只需迭代k次(k为数字中1的个数)
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        while(n):
            count+=1
            n = (n-1)&n # 把最低位的1及其之后置0
        return count
