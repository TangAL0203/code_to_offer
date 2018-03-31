# -*- coding:utf-8 -*-
def FindNumsAppearOnce(array):
    xorValue = AdjacentXor(array) # o(n)
    index = getFirstBit(xorValue) 
    temp1 = 0
    temp2 = 0
    for i in array:   # o(k*n) k为位数
        if isBit1(i, index):
            temp1 = temp1^i
        else:
            temp2 = temp2^i
    return temp1, temp2

# 相邻元素做异或
def AdjacentXor(array):
    if len(array)==0:
        return 0
    else:
        for i in range(len(array)):
            if i==0:
                temp = array[i]
            else:
                temp = temp^array[i]
    return temp

# 获取最低位为1对应的索引，比如0111，则返回-1
# bin(5) = '0b101'
def getFirstBit(num):
    binstr = bin(num)
    lenstr = len(binstr)
    for i in range(-1,-1*(lenstr+1),-1):
        if binstr[i]==str(1):
            return i

def isBit1(num, index):
    binstr = bin(num)
    if binstr[index]==str(1):
        return True
    else:
        return False


# 获取array各个元素中某位为1相加的合数，长度为32位列表
# 第0个元素表示，array中所有元素的二进制表示的最低位的和，依次类推。
# bin(0)='0b0', bin(-1)='-0b1'
# 如果某个特定位上的1加起来，可以被3整除，说明对应x的那位是0
# 如果某个特定位上的1加起来，不可以被3整除，说明对应x的那位是1
# 根据返回的nums1bit来判断只出现一次的整数的值
def get1BitNum(array):
    nums1bit = [0 for i in range(32)]
    for num in array:
        if num<0:
            strlen = len(bin(num))
            for j, bit in enumerate(bin(num)[-1,-strlen+2,-1]):
                if bit==1:
                    nums1bit[j] +=1
        else:
            strlen = len(bin(num))
            for j, bit in enumerate(bin(num)[-1,-strlen+1,-1]):
                if bit==1:
                    nums1bit[j] +=1
    return nums1bit


