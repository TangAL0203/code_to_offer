# -*- coding:utf-8 -*-
g_InvalidInput = False
class Solution:
    def Power(self, base, exponent):
        # write code here
        # 0和0没有意义，需确定是否需要抛出异常
        global g_InvalidInput # 异常处理
        if base==0 and exponent<0:
            g_InvalidInput = True
            return 0
        if exponent==0:
            return 1
        abs_exponent = abs(exponent)
        result = 1
        if exponent<0:
            for i in range(abs_exponent):
                result = result*base
            return 1.0/result
        else:
            for i in range(abs_exponent):
                result = result*base
            return result