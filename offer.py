剑指 Offer 03. 数组中重复的数字
描述: 长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内, 找出重复的数字
思路: 原地交换

class Solution:
    def findRepeatNumber(self, nums: [int]) -> int:
        # hash表法, 定义一个set
        record = set()
        for num in nums:
            if num in record:
                return num
            record.add(num)
        # 原地交换
        for i in range(len(nums)):
            if nums[i] == i:
                continue
            if nums[i] == nums[nums[i]]:
                return nums[i]
            # 位置交换
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1

###############################################################################

剑指 Offer 04. 二维数组中的查找
描述: n*m数组, 左右>增序, 上下>增序, 判断数组是否含有target数
思路: 利用规律, 从左下开始遍历

class Solution(object):
    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        i = 0  # column
        j = len(matrix) - 1  # row
        while j>=0 and i<=(len(matrix[0])-1):
            value = matrix[j][i]
            if target == value:
                return True
            elif target > value:
                i+=1
            else:
                j-=1
        return False

###############################################################################

剑指 Offer 05. 替换空格
描述: 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
思路: 双指针法(两个尾指针), 先for循环确定字符串长度

class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ''
        for xx in s:
            if xx != ' ':
                res += xx
            else:
                res += '%20'
        return res

###############################################################################

剑指 Offer 06. 从尾到头打印链表
描述: 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
思路: 借助堆栈

class Solution(object):
    def reversePrint(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        # 堆栈
        stack = []
        res = []
        node = head
        while node:
            stack.append(node.val)
            node = node.next
        while stack:
            res.append(stack.pop())
        return res

###############################################################################

剑指 Offer 07. 重建二叉树
描述: 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
思路: 递归

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 返回根结点的写法
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        root_index = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:1+root_index], inorder[:root_index])
        root.right = self.buildTree(preorder[1+root_index:], inorder[root_index+1:])
        return root

###############################################################################

剑指 Offer 09. 用两个栈实现队列
描述: 用两个栈实现一个队列。请实现它的两个函数 appendTail 和 deleteHead, 在队列尾部插入整数和在队列头部删除整数的功能。
(若队列中没有元素，deleteHead 操作返回-1)
思路: 两个栈, 一个负责队列入, 一个负责队列出

class CQueue(object):
    def __init__(self):
        # 先创建两个栈
        self.stack1 = []  # 负责入
        self.stack2 = []  # 负责出

    def appendTail(self, value):
        """
        :type value: int
        :rtype: None
        """
        # 入队列就直接入
        self.stack1.append(value)

    def deleteHead(self):
        """
        :rtype: int
        """
        # 分情况考虑
        if self.stack2:
            return self.stack2.pop()
        elif self.stack1:
            # 1出栈 -> 2入栈 -> 2出栈
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        else:
            return -1

###############################################################################

剑指 Offer 10- I. 斐波那契数列
描述：写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
思路：递归

class Solution:
    def fib(self, n: int) -> int:
        # 移位法(循环然后求余)
        a, b = 0, 1  # n-2, n-1
        for _ in range(n):
            # a, b 依次向右移动一位
            a, b = b, a+b  # 先计算a+b, 和b的值，然后再赋值给a, b
        return a % 1000000007

        # # 递归法
        # if n <= 1:
        #     return n
        # return (self.fib(n-1) + self.fib(n-2))

###############################################################################

剑指 Offer 10- II. 青蛙跳台阶问题
描述：一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
思路：fn = fn-1 + fn-2  # 类似于斐波那契数列

class Solution:
    def numWays(self, n: int) -> int:
        # 递归问题: 对于n个台阶, 只能从第n-2或者n-1台阶跳上来, fn = fn-1 + fn-2
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(1, n):
            a, b = b, a+b
        return b % 1000000007

###############################################################################

剑指 Offer 11. 旋转数组的最小数字
描述：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，
输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
思路：二分法(逐渐缩小left和right, 考虑题目数组的性质)

class Solution(object):
    def minArray(self, numbers):
        """
        :type numbers: List[int]
        :rtype: int
        """
        # 找到单调递增的数组即可返回
        left = 0
        right = len(numbers) - 1
        while left < right:
            mid = (left + right) // 2
            # 分情况讨论
            # 1、mid位于第二增区间
            if numbers[mid] < numbers[right]:
                right = mid
            # 2、mid位于第一增区间
            elif numbers[mid] > numbers[right]:
                left = mid + 1
            # 3、重复值存在, 只能逐步减小区间, 不能一次性忽略很多元素
            else:
                right = right - 1

        return numbers[left]

###############################################################################

剑指 Offer 12. 矩阵中的路径
描述：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被
重复使用。(从一个二维字符网格中, 从i,j位置出发, 连线得到的字符串和word匹配就OK)
思路：依次对每一个点做DFS遍历, 采用DFS+回溯的思想

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        # i,j 当前元素在board中的坐标
        # k 当前元素在word中的索引
        # 先访问根节点, 再往下访问, 就是深度优先遍历
        def dfs(i, j, k):
            # 返回false终止条件: i,j没有越界 + 同一个单元格内的字母不允许被重复使用
            # 因为一个位置可以上下左右走, 可能会存在一个结点被重复访问
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]:
                return False
            # 返回true终止条件
            if k == len(word) - 1:
                return True
            board[i][j] = ''  # 该元素置空, 表明已经找过了
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)  # 朝4个方向前进
            board[i][j] = word[k]  # 恢复该元素, 以至于不会影响下一个起始点的遍历
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                # 依次以每个点为起点做DFS遍历
                if dfs(i, j, 0):
                    return True
        return False

###############################################################################

剑指 Offer 14- I. 剪绳子
描述：长度为n绳子, 剪成m段, 求这m段绳子长度的乘积最大值是多少?
思路：1、通过数论推导, 等分最大, 求极值点, 得到等分为3最大. 2、i-1可以推出i, 用动态规划思想

# 参考: https://leetcode-cn.com/problems/jian-sheng-zi-lcof/solution/mian-shi-ti-14-i-jian-sheng-zi-tan-xin-si-xiang-by/
class Solution:
    def cuttingRope(self, n: int) -> int:
        ## 数论法: 1、等分乘积最大。2、设等分长度为x, 乘积用x表示, 求极值点, 得到x==3最大；
        # 结论：尽可能将绳子以长度3等分为多段时，乘积最大
        # if n<=3:
        #     # 必须切一刀
        #     return n-1
        # temp = n % 3
        # if temp == 0:
        #     return 3 ** (int(n//3))
        # elif temp == 1:
        #     # 将1+3转为2x2
        #     return 3 ** (int(n//3)-1) * 4
        # else:
        #     return 3 ** (int(n//3)) * 2

        ## 动态规划, 参考: https://leetcode-cn.com/problems/jian-sheng-zi-lcof/solution/jian-zhi-offer-14-i-jian-sheng-zi-huan-s-xopj/
        # 长度为i的绳子, 可以由长度为i-1的结果推导出来.
        dp = [0] * (n+1)  # 多了长度为0
        dp[2] = 1
        for i in range(3, n+1):
            for j in range(2, i):
                dp[i] = max(dp[i], (j * (i-j)), j * dp[i-j])
        return dp[n]

###############################################################################

剑指 Offer 15. 二进制中1的个数
描述：输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。
思路：循环, 移位, 累加1的个数

class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += n & 1
            n >>= 1
        return res

###############################################################################

剑指 Offer 16. 数值的整数次方
描述：实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。
思路：将n转为二进制数, n<0通过对偶转为n>0的解法, 二分法确定n的2进制位数

# 答案参考: https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/solution/mian-shi-ti-16-shu-zhi-de-zheng-shu-ci-fang-kuai-s/
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0
        res = 1
        if n < 0:
            x, n = 1 / x, -n  # 转换为 n > 0 的情况
        # 假如n=8, 1000, 前三个都是0, 右移到1的时候才乘到结果中去
        # 将n变为二进制位, 然后用二分法来做
        while n:
            # 按位与
            if n & 1:
                res *= x  # 为奇数才会乘到结果中去
            x *= x  # 依次计算x^1, x^2, x^4, ..., x^(m-1)
            n >>= 1  # 右移除2
        return res

###############################################################################

剑指 Offer 17. 打印从1到最大的n位数
描述：输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
思路：要考虑大数问题, 用字符串来处理, 采用dfs进行遍历得到n个数的排列(0001要去掉000, 这个时候要有一个start记录起始点)

class Solution:
    def printNumbers(self, n: int) -> List[int]:
        ## 大数打印, 处理越界问题, 需要返回字符串才能处理大数打印问题
        # 深度优先遍历, 找全排列
        def dfs(x):
            # 终止条件
            if x == n:
                s = ''.join(num[self.start:])
                if s != '0':
                    res.append(int(s))
                # 进位
                if (n - self.start) == self.nine:
                    self.start -= 1
                return
            for i in range(10):
                if i == 9:
                    self.nine += 1
                num[x] = str(i)
                dfs(x+1)
            self.nine -= 1  # 回溯

        num, res = ['0'] * n, []
        self.nine = 0
        self.start = n - 1
        dfs(0)
        return res

###############################################################################

剑指 Offer 18. 删除链表的节点
描述：给定一个单链表 和 要删除的结点的值，返回删除该结点后链表的头结点
思路：定义删除结点前的一个前置结点pre, 依次更新pre和cur, pre落后cur一个结点

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def deleteNode(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # 边界值判断
        if head is None:
            return head
        if head.val == val:
            return head.next
        pre = None  # 记录当前head的前一个结点，用来做删除操作
        cur = head
        while cur.val != val:
            pre = cur
            cur = cur.next
        pre.next = cur.next

        return head

###############################################################################

剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
描述：输入一个整数数组，调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
思路：双指针法, 左指针和右指针为偶数和奇数, 交换两个数字的顺序

class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        #  双指针法
        left = 0
        right = len(nums) - 1
        while left < right:
            # 如果left和right为偶数和奇数, 交换两者位置
            if (not nums[left] % 2) and (nums[right] % 2):
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            if (nums[left] % 2):
                # left为奇数, left+=1
                left += 1
            if (not nums[right] % 2):
                # right为偶数, right-=1
                right -= 1
        return nums

###############################################################################

剑指 Offer 22. 链表中倒数第k个节点
描述：输入一个链表，输出该链表中倒数第k个节点。链表的尾节点是倒数第1个节点。
思路：双指针法, 设置两个快慢指针

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getKthFromEnd(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        # 难点在于不知道链表的长度，但只想遍历一次，所以用双指针，快慢指针
        slow  = head
        fast = head
        # 快指针先走k步数, 当快指针为None时候，返回慢指针
        for _ in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        return slow

###############################################################################

剑指 Offer 24. 反转链表
描述：定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
思路：递归法 + 正常遍历

class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 递归解法
        # 终止条件
        if head is None or head.next is None:
            return head
        node = self.reverseList(head.next)
        # 将head和head.next反转, 这个时候head的next还是指向node反转链表的最后一个元素, 需要做一下反转
        head.next.next = head
        head.next = None
        return node

        # 正常遍历解法
        pre = None  # 记录当前的翻转链表的头结点
        cur = head
        while cur:
            next = cur.next  # 暂存下一个结点
            cur.next = pre  # 将cur结点更新为翻转头结点
            pre = cur
            cur = next  # 接着去下一个结点
        return pre

###############################################################################

剑指 Offer 25. 合并两个排序的链表
描述：输入两个递增的链表, 合并这两个链表, 确保合并后的链表也是递增的
思路：递归解法

# 递归操作
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

###############################################################################

剑指 Offer 27. 二叉树的镜像
描述：输入一个二叉树，输出它的镜像
思路：借助辅助栈(层次优先遍历) or 递归

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        ## 辅助栈法(层次优先遍历), 自上而下交换
        if not root:
            return
        stack = [root]
        while stack:
            node = stack.pop()
            # 这里left先入, right后入
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            # 交换left和right
            node.left, node.right = node.right, node.left
        return root

        # 递归方法, 自下而上交换
        if root is None:
            return root
        temp = root.left
        root.left = self.mirrorTree(root.right)  # 镜像树中，左子树为右子树的镜像
        root.right = self.mirrorTree(temp)  # 镜像树中，右子树为左子树的镜像
        return root

###############################################################################

剑指 Offer 28. 对称的二叉树
描述：请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
思路：对左子树和右子树做遍历, 确保每一个节点都相等

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def match(L, R):
            # 先做非None判断, 再check val值相等
            if L is None and R is None:
                return True  # 都为None, 则全部都匹配上, 返回True
            if L is None and R is not None:
                return False
            elif L is not None and R is None:
                return False
            else:
                return (L.val == R.val) and match(L.left, R.right) and match(L.right, R.left)

        return match(root.left, root.right) if root else True

###############################################################################

剑指 Offer 29. 顺时针打印矩阵
描述：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
思路：用字典记录状态转移矩阵

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        # 用字典记录转移矩阵
        change = {
            'shang': 'you',
            'you': 'xia',
            'xia': 'zuo',
            'zuo': 'shang',
        }
        state = 'shang'
        res = []
        # 易错地方: 二维list需要判断matrix和matrix[0]都不为空
        while matrix and matrix[0]:
            if state == 'shang':
                res += (matrix.pop(0))
            elif state == 'you':
                res += ([xx.pop() for xx in matrix])
            elif state == 'xia':
                res += (matrix.pop()[::-1])
            else:
                res += ([xx.pop(0) for xx in matrix][::-1])
            state = change[state]
        return res

###############################################################################

剑指 Offer 30. 包含min函数的栈
描述：实现包含min函数的栈, 在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)
思路：栈的实现借助list, min操作借助一个辅助栈min_stack来实现

class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.min_stack:
            self.min_stack.append(x)
        else:
            if x <= self.min_stack[-1]:
                self.min_stack.append(x)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.min_stack[-1]

###############################################################################

剑指 Offer 31. 栈的压入、弹出序列
描述：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
思路：借用一个辅助栈 stack ，模拟 压入 / 弹出操作的排列。根据是否模拟成功，即可得到结果。

class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        # 借助辅助栈, 模拟压入弹出过程
        stack, i = [], 0  # i记录出栈元素的顺序
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop() # 模拟出栈
                i += 1
        return not stack

###############################################################################

剑指 Offer 32 - I. 从上到下打印二叉树
描述：从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
思路：借助队列实现

# 通过队列, 将每层结点按照从左往右的顺序添加进去, 然后进行访问
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        seq = [root]
        while seq:
            node = seq.pop(0)
            res += [node.val]
            if node.left:
                seq.append(node.left)
            if node.right:
                seq.append(node.right)
        return res

###############################################################################

剑指 Offer 32 - II. 从上到下打印二叉树 II
描述：从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
思路：借助队列实现

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        seq = [root]
        while seq:
            temp = []
            next_seq = []
            for node in seq:
                temp.append(node.val)
                # 左右顺序: 先添加左结点, 再添加右结点
                if node.left:
                    next_seq.append(node.left)
                if node.right:
                    next_seq.append(node.right)
            res.append(temp)
            # 更新seq
            seq = next_seq
        return res

###############################################################################

剑指 Offer 32 - III. 从上到下打印二叉树 III
描述：Z子形打印二叉树, 即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
思路：在上面的基础上加一个奇偶判断即可

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        seq = [root]
        i = 0
        while seq:
            temp = []
            next_seq = []
            for node in seq:
                temp.append(node.val)
                if node.left:
                    next_seq.append(node.left)
                if node.right:
                    next_seq.append(node.right)
            if i % 2 == 0:
                # 从左到右打印
                res.append(temp)
            else:
                # 从右到左打印
                res.append(temp[::-1])
            i += 1
            seq = next_seq
        return res

###############################################################################

剑指 Offer 33. 二叉搜索树的后序遍历序列
描述：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回true，否则返回false。假设输入的数组的任意两个数字都互不相同。
思路：二叉搜索树的子树也是二叉搜索树, 后序遍历可以找到左, 右, 根, 然后做递归

class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:

        def recur(i, j):
            # i, j表示postorder子区间左右的索引
            if i >= j:
                return True
            temp = i
            while postorder[temp] < postorder[j]:
                temp += 1
            # 找到右子树的第一个节点, 继续遍历, 直到到达末尾
            right = temp
            while postorder[temp] > postorder[j]:
                temp += 1
            return temp == j and recur(i, right-1) and recur(right, j-1)

        return recur(0, len(postorder)-1)

###############################################################################

剑指 Offer 34. 二叉树中和为某一值的路径
描述：输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
思路：dfs+回溯法, res记录结果, path记录路径, path.pop()做回溯

import copy
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        # 回溯方法解答: 通过试错, 找到所有问题的答案
        # path 记录当前可能的结果, res记录所有的结果
        res, path = [], []
        def recur(root, tar):
            if not root:
                return
            path.append(root.val)
            tar -= root.val
            if tar == 0 and not root.left and not root.right:
                res.append(copy.deepcopy(path))
            recur(root.left, tar)
            recur(root.right, tar)
            path.pop()  # 回溯，撤销处理结果。可能不会立即执行, 但是会被记住。
        recur(root, target)
        return res

###############################################################################

剑指 Offer 35. 复杂链表的复制
描述：请实现copyRandomList函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个next指针指向下一个节点，还有一个random指针指向链表中的任意节点或者null。
思路：借助字典, 返回字典中head值对应的新head

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # 借助字典
        if not head:
            return head
        dic = {}
        cur = head
        # 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while cur:
            dic[cur] = Node(cur.val)  # 先构建节点, next和random都先不管
            cur = cur.next
        # 构建新节点的 next 和 random 指向
        cur = head
        while cur:
            dic[cur].next = dic.get(cur.next, None)
            dic[cur].random = dic.get(cur.random, None)
            cur = cur.next
        # 返回新链表的头节点
        return dic[head]

###############################################################################

剑指 Offer 38. 字符串的排列
描述：输入一个字符串，打印出该字符串中字符的所有排列。
思路：全排列的问题, 考虑用DFS+回溯+剪枝来做。(注意不字符串中不能出现重复字符, 当一个字符已经被访问时, 下次不能再访问)

class Solution:
    def permutation(self, s: str) -> List[str]:
        c_list, res = list(s), []
        def backtrack(x):
            if x == (len(s) - 1):
                res.append(''.join(c_list))
                return
            temp = []
            for i in range(x, len(s)):
                # 该层已经取过的元素, 不能再取了
                if c_list[i] in temp:
                    continue
                temp.append(c_list[i])
                c_list[x], c_list[i] = c_list[i], c_list[x]  # 交换i, x位置, 模拟排列的过程(固定第k位), x代表选定的位置, 剩余的表示待挑选的
                backtrack(x+1)
                c_list[x], c_list[i] = c_list[i], c_list[x]  # 回溯
        backtrack(0)
        return res

###############################################################################

剑指 Offer 39. 数组中出现次数超过一半的数字
描述：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
思路：摩尔投票法, 创建两个变量 ans, count, 表示当前领先者和票数, 票数为0更新领先者

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 时间复杂度: O(n), 空间复杂度: O(1)
        ans, count = nums[0], 1
        for num in nums[1:]:
            if num != ans:
                count -= 1
            else:
                count += 1
            # 更新领先者
            if count == 0:
                ans = num
                count = 1

        return ans

###############################################################################

剑指 Offer 40. 最小的k个数
描述：输入整数数组arr，找出其中最小的k个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
思路：由于该数组没有任何特点, 写一个快排来解决

class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 写一个快排算法,
        def quick_sort(nums, left, right):
            if left >= right:
                return
            low = left
            high = right
            key = nums[left]
            while left < right:
                # 让nums左边都比key小, 右边都比key大
                while left < right and nums[right] >= key:
                    right -= 1
                nums[left] = nums[right]  # 较小值移到左边
                while left < right and nums[left] <= key:
                    left += 1
                nums[right] = nums[left]
            # 将key值归位
            nums[right] = key
            # 分治思想, 递归调用
            quick_sort(nums, low, left-1)
            quick_sort(nums, left+1, high)
            return nums

        list = quick_sort(arr, 0, len(arr)-1)
        return list[:k]

###############################################################################

剑指 Offer 42. 连续子数组的最大和
描述：输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
思路：不能用双指针法(因为子序列可能断开), 构建两个变量res和cur_max, res为最终结果, cur_max记录当前的最大值

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur_max = nums[0]
        res = nums[0]
        for i in range(1, len(nums)):
            cur_max = max(cur_max, 0) + nums[i]
            res = max(res, cur_max)
        return res

###############################################################################

剑指 Offer 43. 1～n 整数中 1 出现的次数
描述：输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
思路：数位DP, 统计每个位置上1出现的个数, 累加起来(当前值分三种情况讨论: 0, 1, 2-9)

# 数位dp: res = 各个位置上1总数之和
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0
        while high != 0 or cur != 0:
            if cur == 0:
                res += high * digit
            elif cur == 1:
                res += high * digit + low + 1
            else:
                res += (high + 1) * digit
            low += cur * digit  # 更新low值
            cur = high % 10  # cur左移一位
            high //= 10  # high左移一位
            digit *= 10  # 更新数位
        return res

###############################################################################

剑指 Offer 45. 把数组排成最小的数
描述：输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
思路：要找最小的, 则所有的数字要做一个升序排序, 所以用快排即可

# 采用排序: 先把数字数组转为字符串数组, 然后用快排排序, 最后把str_list给join起来返回
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def quick_sort(nums, left, right):
            if left >= right:
                return
            low = left
            high = right
            val = nums[left]
            while left < right:
                # right都比val大
                while left < right and ((nums[right]+val) >= (val + nums[right])):
                    right -= 1
                nums[left] = nums[right]
                # left都比val小
                while left < right and ((nums[left]+ val) <= val + nums[left]):
                    left += 1
                nums[right] = nums[left]
            nums[left] = val
            quick_sort(nums, low, left-1)
            quick_sort(nums, left+1, high)
        temp_nums = [str(xx) for xx in nums]
        quick_sort(temp_nums, 0, len(nums)-1)
        return ''.join(temp_nums)

###############################################################################

剑指 Offer 46. 把数字翻译成字符串
描述：给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。
思路：注意12有两种翻译法, 动态规划, 推出转移矩阵, fn=fn-1+fn-2 if xn和xn-1可以翻译 else = xn-1 when xn和xn-1不可翻译

class Solution:
    def translateNum(self, num: int) -> int:
        # 动态规划: 推出转移矩阵, fn = fn-1 + fn-2 if xn和xn-1可以翻译 else = xn-1 when xn和xn-1不可翻译
        s = str(num)
        dp0, dp1 = 1, 1  # n-2, n-1
        for i in range(2, len(s)+1):
            temp = s[i-2: i]
            dp2 = dp0 + dp1 if "10" <= temp <= "25" else dp1  # 转移公式
            # 更新转移公式的元素
            dp0, dp1 = dp1, dp2

        return dp1 # 最终dp1==dp2, 这里返回dp1, 怕dp2变量没有创建

###############################################################################

剑指 Offer 48. 最长不含重复字符的子字符串
描述：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
思路：滑动窗口, 用dic记录字符的最新索引位置, 更新left, 移动right

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 滑动窗法, dic记录字符的最近索引
        dic, res, left = {}, 0, -1
        for right in range(len(s)):
            if s[right] in dic:
                # 更新left
                left = max(dic[s[right]], left)  # dic[s[right]]可能落后于left, 比如: 'arabcar'
            dic[s[right]] = right
            res = max(res, (right-left))
        return res

###############################################################################

剑指 Offer 50. 第一个只出现一次的字符 - 简单
描述：在字符串s中找出第一个只出现一次的字符。如果没有，返回一个单空格。s只包含小写字母。
思路：借助字典, 记录每个字符出现的次数

from collections import OrderedDict
class Solution:
    def firstUniqChar(self, s: str) -> str:
        # 最多有 26 个不同字符，HashMap 存储需占用 O(26)=O(1) 的额外空间
        hash_dict = OrderedDict()
        for xx in s:
            if xx not in hash_dict:
                hash_dict[xx] = 1
            else:
                hash_dict[xx] += 1
        for key, value in hash_dict.items():
            if value == 1:
                return key
        return ' '

###############################################################################

剑指 Offer 52. 两个链表的第一个公共节点
描述：输入两个链表，找出它们的第一个公共节点。
思路：交替遍历, 当相等时返回节点

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        ## 方法一：去你的世界纠缠你
        curA = headA
        curB = headB
        while curA != curB:
            # 不会死循环，就算没有相交的点，最后curA和curB都是None，也会跳出该循环
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA
        return curA

###############################################################################

剑指 Offer 53 - I. 在排序数组中查找数字 I
描述：
思路：二分法先找到该数字, 然后在该数字的左右区间做for循环遍历

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        n = len(nums)
        left = 0
        right = len(nums) - 1
        count = 0
        while left <= right:
            mid = (left+right)//2
            if nums[mid] > target:
                right = mid-1
            elif nums[mid] < target:
                left = mid+1
            else:
                count += 1
                # 遍历右边
                temp = mid + 1
                while temp<n:
                    if nums[temp] == target:
                        count += 1
                    temp += 1
                # 遍历左边
                temp = mid - 1
                while temp >= 0:
                    if nums[temp] == target:
                        count += 1
                    temp -= 1
                break
        return count

###############################################################################

剑指 Offer 53 - II. 0～n-1中缺失的数字
描述：一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
思路：该数字是有序的, 考虑二分法, mid == nums[mid] 去缩小范围

class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        while left <= right:
            # 判断哪个区间可能存在该数字
            mid = (left+right) // 2
            if mid != nums[mid]:
                right = mid-1
            else:
                left = mid+1

        return left  # [0,1,3]举例去想

###############################################################################

剑指 Offer 54. 二叉搜索树的第k大节点
描述：给定一棵二叉搜索树，请找出其中第k大的节点。
思路：对二叉树做(右, 根, 左)遍历, 这样得到的是一个倒序序列, 用一个全局遍历来记录倒到哪里了

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        # 二叉搜索树特点: 左子树小于根结点, 右子树大于根结点
        # 中序遍历(左, 根, 右)为升序, (右, 根, 左)为倒序
        def recur(root):
            if not root:
                return
            recur(root.right)  # 访问右子树
            self.k -= 1  # 必须写在访问右子树的后面(当右子树为None时候, 访问了root才有意义)
            if self.k == 0:
                self.res = root.val
                return  # 找到该值, 提前返回
            recur(root.left)  # 访问左子树

        self.k = k  # 类变量来计数
        recur(root)
        return self.res

###############################################################################

剑指 Offer 55 - I. 二叉树的深度
描述：输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
思路：递归

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 递归
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

###############################################################################

剑指 Offer 55 - II. 平衡二叉树
描述：输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。(具有递归性质)
思路：后序遍历, 依次判断左右子树是否为平衡树, 递归函数返回树的深度 or -1(表示子树不是平衡树)

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def recur(root):
            if not root:
                return 0
            left = recur(root.left)
            if left == -1:
                return -1
            right = recur(root.right)
            if right == -1:
                return -1
            return max(left, right) + 1 if abs(left - right) <= 1 else -1

        return recur(root) != -1

###############################################################################

剑指 Offer 56 - I. 数组中数字出现的次数
描述：一个整型数组nums里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
思路：分组异或, 用异或来处理, 出现两次的数字, 异或操作之后为0, 先全部做异或, 再将该数组分为两堆

class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        def xnor_reduce(nums):
            if len(nums) == 1:
                return nums[0]
            return nums[0] ^ xnor_reduce(nums[1:])

        xnor_value = xnor_reduce(nums)
        div = 1
        while not (div & xnor_value):
            div <<= 1 # 找到xnor_value为1的位数
        nums_0 = None
        nums_1 = None
        for num in nums:
            if num & div:
                if nums_1 is None:
                    nums_1 = num
                else:
                    nums_1 = nums_1 ^ num
            else:
                if nums_0 is None:
                    nums_0 = num
                else:
                    nums_0 = nums_0 ^ num

        return [nums_0, nums_1]

###############################################################################

剑指 Offer 57. 和为s的两个数字
描述：输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
思路：双指针法, 头尾指针

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        start = 0
        end = len(nums) - 1
        while end > start:
            if (nums[start] + nums[end]) > target:
                end -= 1
            elif (nums[start] + nums[end]) < target:
                start += 1
            else:
                return [nums[start], nums[end]]

###############################################################################

剑指 Offer 57 - II. 和为s的连续正数序列
描述：输入一个正整数target，输出所有和为target的连续正整数序列（至少含有两个数）。
思路：双指针法, 表示连续正整数序列的左右区间, 用一个变量s记录累加和(区间移动时, 更新变量s)

class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        ## 双指针法, 只不过left, right的初始值为1, 2
        left, right, s, res = 1, 2, 3, []
        while left < right:
            if s == target:
                res.append(list(range(left, right+1)))
                # 当前序列和已经==target, 则指针移动方向为缩小序列范围
                s -= left
                left += 1
            elif s > target:
                # 缩小序列范围
                s -= left
                left += 1
            else:
                # 右移right
                right += 1
                s += right
        return res

###############################################################################

剑指 Offer 58 - I. 翻转单词顺序
描述：输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
思路：双指针法, 从尾部开始遍历

class Solution:
    def reverseWords(self, s: str) -> str:
        # 双指针法, 从尾部开始左遍历
        s = s.strip()
        left, right, res = len(s)-1, len(s)-1, []
        while left >= 0:
            while left >= 0 and s[left] != ' ':
                left -= 1
            res.append(s[left+1:right+1])
            while left >= 0 and s[left] == ' ':
                left -= 1
            right = left

        return ' '.join(res)

###############################################################################

剑指 Offer 58 - II. 左旋转字符串
描述：字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
思路：三次翻转 or 申请空间 or 切片操作

class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        # O(N) 复杂度
        return s[n:] + s[:n]

###############################################################################

剑指 Offer 59 - II. 队列的最大值
描述：请定义一个队列并实现函数max_value得到队列里的最大值，要求函数max_value、push_back和pop_front的均摊时间复杂度都是O(1)。若队列为空，pop_front和max_value需要返回-1
思路：借助双向队列, 更新0元素(代表局部队列的一个最大值), 尾部元素代表整个队列最大值

class MaxQueue(object):
    ## 借助辅助栈
    def __init__(self):
        self.list = []
        self.max_stack = []  # 借助双向队列, 头部入+出, 尾部出

    def max_value(self):
        """
        :rtype: int
        """
        return self.max_stack[-1] if self.max_stack else -1

    def push_back(self, value):
        """
        :type value: int
        :rtype: None
        """
        # 更新最大值(后入的元素, 会改变之前的最大值)
        while self.max_stack and self.max_stack[0] < value:
            self.max_stack.pop(0)
        self.max_stack.insert(0, value)  # 注意不是append, max_stack元素记录的是队列左边的最大值
        self.list.append(value)

    def pop_front(self):
        """
        :rtype: int
        """
        if self.list:
            x = self.list.pop(0)
            if x == self.max_stack[-1]:
                self.max_stack.pop()
            return x
        else:
            return -1

###############################################################################

剑指 Offer 60. n个骰子的点数, 概率情况分布
描述：把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
思路：动态规划来做

class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        ## 解法二: 用动态规划
        # 递推公式1: 逆推(fn <= fn-1); 递推公式2: 正推(fn-1 => fn); 此题用正推不用处理越界的问题
        dp = [1/6] * 6  # f(i-1)
        for i in range(2, n+1):
            tmp = [0,] * (5*i + 1)  # f(i)
            for j in range(len(dp)):
                for k in range(6):
                    tmp[j+k] += ((dp[j]) * 1/6)
            # 更新dp
            dp = tmp
        return dp

###############################################################################

剑指 Offer 61. 扑克牌中的顺子
描述：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
思路：智商题, 先排除有重复数字情况, 没有重复数字时, (max - min) <= 4即可; 可在坐标轴画出可能

class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        non_zero_nums = []
        for xx in nums:
            if xx != 0:
                non_zero_nums.append(xx)
        # 首先不能有重复数字
        if len(set(non_zero_nums)) != len(non_zero_nums):
            return False
        else:
            # 当不含有重复数字时, 只要max-min <= 4即可
            return (max(non_zero_nums) - min(non_zero_nums)) <= (4)

###############################################################################

剑指 Offer 62. 圆圈中最后剩下的数字
描述：0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
思路：寻找递推公式(有点难)

# 本质: 寻找递推公式
# f(n,m) = 0 if n==1 else [f(n-1,m)+m] % n # 最后存活人的位置为0

class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        ans = 0
        for i in range(2, n+1):
            ans = (ans + m) % i  # 每次循环右移m个元素, 序列长度更新为i
        return ans

###############################################################################

剑指 Offer 63. 股票的最大利润
描述：假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
思路：记录股票当前最小值, 更新最大利润

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        cur_min = prices[0]
        res = 0
        for price in prices:
            res = max(res, price-cur_min)
            if price < cur_min:
                cur_min = price
        return res

###############################################################################

剑指 Offer 64. 求1+2+…+n
描述：求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
思路：用and的短路思想去做递归

class Solution:
    def __init__(self):
        self.res = 0
    def sumNums(self, n: int) -> int:
        # 用递归, 但是不用if去判断终止条件. 用短路思想: and 逻辑
        n > 1 and self.sumNums(n-1)
        self.res += n
        return self.res


剑指 Offer 67. 把字符串转换成整数
描述：写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。
思路：用res, i, sign 三个变量表示结果，遍历起始位置，符号位

class Solution:
    def strToInt(self, str: str) -> int:
        str = str.strip()                      # 删除首尾空格(删除首尾连续的空格)  '   123'.strip() > '123'
        if not str:
            return 0                   # 字符串为空则直接返回
        res, i, sign = 0, 1, 1  # i表示开始拼接的索引, 默认第一个字符为符号位, 且为正符号位
        int_max, int_min, bndry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        if str[0] == '-':
            sign = -1            # 保存负号
        elif str[0] != '+':
            i = 0              # 若无符号位，则需从 i = 0 开始数字拼接
        for c in str[i:]:
            if not '0' <= c <= '9' :
                break     # 遇到非数字的字符则跳出
            res = 10 * res + ord(c) - ord('0') # 数字拼接(从左往右拼接, 最先拼接的, 后面x10的次数就越多)
            # 数字越界处理
            if sign == 1:
                if res > int_max:
                    return int_max
            else:
                if -res < int_min:
                    return int_min
        return sign * res

###############################################################################

剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
描述：给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
思路：递归

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ### 找树的最近公共祖先
        ## 方法一: 迭代法
        # if p.val > q.val:
        #     p, q = q, p  # 保证 p.val < q.val
        # while root:
        #     if root.val < p.val and root.val < q.val: # p,q 都在 root 的右子树中
        #         root = root.right
        #     elif root.val > q.val: # p,q 都在 root 的左子树中
        #         root = root.left
        #     else:
        #         # 满足root为p, q的条件: 1、root等于p或q; 2、p和q位于root的异侧
        #         break
        # return root

        ## 方法二: 递归
        if root.val < p.val and root.val < q.val:
            # 去右子树找
            return self.lowestCommonAncestor(root.right, p, q)
        if root.val > p.val and root.val > q.val:
            # 去左子树找
            return self.lowestCommonAncestor(root.left, p, q)
        return root

###############################################################################

剑指 Offer 68 - II. 二叉树的最近公共祖先
描述：给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
思路：递归

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # 后序遍历
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)  # 左子树遍历到p或q
        right = self.lowestCommonAncestor(root.right, p, q)  # 右子树遍历到p或q
        if not left and not right:  # 左右子树都不含有p, q
            return None # 1.
        if not left:
            return right # 3.
        if not right:
            return left # 4.
        return root # 2. if left and right:、

###############################################################################
