#-*-coding:utf-8-*-

# 三种操作：插入，删除，替换
# out[i][j] 表示长度为i的字符串转换为长度为j的字符串需要进行多少次操作。
# i,j可以由(i-1,j-1)，(i,j-1)，(i-1,j)得到
# (i-1,j-1) => (i,j) 可修改，可不变，取决于srtStr[i]是否等于distStr[j]
# (i,j-1) => (i,j) 可以插入得到
# (i-1,j) => (i,j) 可以删除得到
def getEditDistance(srtStr, distStr):
    M = len(srtStr)
    N = len(distStr)
    out = [[0]*(N+1) for _ in range(M+1)]
    for i in range(M+1):
        for j in range(N+1):
            if i==0 and j==0:
                out[i][j] = 0
            elif i==0 and j!=0:
                out[i][j] = j  # 空字符串->长度为j的目的字符串   插入操作
            elif i!=0 and j==0:
                out[i][j] = i # 长度为i的源字符串->空字符串  删除操作
            else:
                if srtStr[i-1]==distStr[j-1]:
                    out[i][j] = min(out[i-1][j-1],out[i][j-1]+1,out[i-1][j]+1)
                else:
                    out[i][j] = min(out[i-1][j-1]+1,out[i][j-1]+1,out[i-1][j]+1)
    return out[M][N]

srcTest = 'kitten'
distTest = 'sitting'
print getEditDistance(srcTest, distTest)

# 二种操作：插入，删除，替换
def getEditDistance(srtStr, distStr):
    M = len(srtStr)
    N = len(distStr)
    out = [[0]*(N+1) for _ in range(M+1)]
    for i in range(M+1):
        for j in range(N+1):
            if i==0 and j==0:
                out[i][j] = 0
            elif i==0 and j!=0:
                out[i][j] = j  # 空字符串->长度为j的目的字符串   插入操作
            elif i!=0 and j==0:
                out[i][j] = i # 长度为i的源字符串->空字符串  删除操作
            else:
                if srtStr[i-1]==distStr[j-1]:
                    out[i][j] = min(out[i-1][j-1],out[i][j-1]+1,out[i-1][j]+1)
                else:
                    out[i][j] = min(out[i-1][j-1]+2,out[i][j-1]+1,out[i-1][j]+1)
    return out[M][N]

srcTest = 'kitten'
distTest = 'sitting'
print getEditDistance(srcTest, distTest)