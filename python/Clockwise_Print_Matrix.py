# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    # 分解为四个步骤，注意判断边界条件。
    def printMatrix(self, matrix):
        # write code here
        result = []
        while matrix:
            result = result+matrix.pop(0) # 取出第一行
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix) # 矩阵逆旋转

        return result

    # [[1,2,3],[4,5,6]] => [[3,6],[2,5],[1,4]]
    # 将矩阵逆时针旋转
    def turn(self,matrix):
            num_r = len(matrix)
            num_c = len(matrix[0])
            newmat = []
            for j in range(num_c-1,-1,-1):
                newmat2 = []
                for i in range(num_r):
                    newmat2.append(matrix[i][j])
                newmat.append(newmat2)
            return newmat

            