#-*-coding:utf-8-*-
####  大数相加  ####
import os

# s1 = '12578846314884'
# s2 = '1526354'

# s1 = '1526354'
# s2 = '12578846314884'

s1 = '154567154567154567154567154567154567'
s2 = '12345678791234567879123456787912345678791234567879'

# s1 = '154567'
# s2 = '1234567879'

res = ''

print s1
print s2
if len(s1)>=len(s2):
    s1 = s1
    s2 = s2
else:
    temp = s1
    s1 = s2
    s2 = temp

sameLen = min(len(s1),len(s2))
carryNum = 0  # 进位数字
nowNum = 0  # 保留的数字
for i in range(sameLen-1,-1,-1):
    tempSum = int(s1[i+(len(s1)-len(s2))]) + int(s2[i]) + carryNum
    if tempSum>=10:
        carryNum = tempSum//10
        nowNum = tempSum-10
    else:
        carryNum = 0
        nowNum = tempSum
    res+=str(nowNum)
if (len(s1)-len(s2))>=1:
    for i in range(len(s1)-len(s2)-1,-1,-1):
        tempSum = int(s1[i])+carryNum
        if tempSum>=10:
            carryNum = tempSum//10
            nowNum = tempSum-10
        else:
            carryNum = 0
            nowNum = tempSum
        res+=str(nowNum)
    if carryNum>0:
        res+=str(carryNum)
else:
    if carryNum>0:
        res+=str(carryNum)
print res[::-1]
print int(res[::-1])==(int(s1)+int(s2))