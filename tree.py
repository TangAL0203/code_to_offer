class Solution:
    # 中序
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return [] 
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    # 前序
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return [] 
        return  [root.val] + self.inorderTraversal(root.left) + self.inorderTraversal(root.right)

    # 后序
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return [] 
        return  self.inorderTraversal(root.left) + self.inorderTraversal(root.right) + [root.val]


# 非递归

class Solution:
    # 前序
    def preorderTraversal(self, root):
        ret, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                ret.append(node.val)
                #注意压入栈的顺序,先压入右孩子，再压入左孩子
                stack.append(node.right)
                stack.append(node.left)
        return ret          

    # 中序
    def inorderTraversal(self, root):
        ret, stack = [], []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                temNode = stack.pop()
                ret.append(temNode.val)
                root = temNode.right
        return ret
    # 后序
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ret, stack = [], []
        while root or stack:
            if root:
                stack.append(root)
                ret.insert(0, root.val)
                root = root.right
            else:
                node = stack.pop()
                root = node.left
        return ret

## 二叉树的层序遍历

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import deque
        ## 返回: list[list, 每层单独放一个list]
        if not root:
            return []
        # 初始化
        res = []
        queue = deque([root])
        ## 队列来做
        while queue:
            ans = []  # 当前层的结果
            for _ in range(len(queue)):
                node = queue.popleft()
                ans.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(ans)
        return res


