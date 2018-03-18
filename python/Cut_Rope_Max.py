# -*- coding:utf-8 -*-
class Solution:
    def Get_Max_Product(self, number):
        # 当n>=5时，3(n-3)>=2(n-2)>n，这个时候可以剪绳子。
        # 当n<5时，没必要剪绳子，但当n=4时，要求必须简一刀绳子，可以剪成2x2
        if number<=1:
            return 0
        if number==2:
            return 1
        if number==3:
            return 2
        TimesOf3 = (number/3) # 330 331  两种情况
        if number-(TimesOf3*3)==1:
            TimesOf3-=1
        TimesOf2 = (number-(TimesOf3*3))/2

        return int(3**TimesOf3)*(2**TimesOf2)

