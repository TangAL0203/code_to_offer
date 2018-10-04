#-*-coding:utf-8-*-
import os
Total_W = 10
Total_N = 4

P_List = [100,40,30,50]
W_List = [5,4,6,3]

# 参考：https://www.nowcoder.com/discuss/3574
# 维护一个数组
# 局部解 -> 全局解
# 5*10的矩阵，行值表示取前几个物体，列值表示背包的容量，数组中的元素表示N个物体，最大容量为K的背包可以携带的最大价值
N_Weights = []
for i in range(Total_N):
    N_Weights.append([0]*(Total_W+1))


for i in range(Total_N):
    if i==0:
        for j in range(1,Total_W+1):
            if j>=W_List[i]:
                N_Weights[i][j] = P_List[i]
    else:
        for j in range(1,Total_W+1):
            if j<W_List[i]:
                N_Weights[i][j] = N_Weights[i-1][j]
            else:
                N_Weights[i][j] = max(N_Weights[i-1][j-W_List[i]]+P_List[i], N_Weights[i-1][j])

print("max value is: {}".format(N_Weights[-1][-1])) # 2D数组的最后一个元素即是最大价值

# 找到哪些物品放入到了背包中
in_Indexs = [] # 存放放入背包的物品的序号，回溯法
curWeight = Total_W
for i in range(Total_N-1,0,-1):
    if N_Weights[i][curWeight]>N_Weights[i-1][curWeight]:
        in_Indexs.append(i) # 将物品放入背包
        curWeight = curWeight-W_List[i]
    else:
        if i==1:
            in_Indexs.append(0) # 判断是否要加上第0个元素
        pass

print("in package indexs is: {}".format(in_Indexs))