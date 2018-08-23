# python 
# 1. 创建二叉树
# 2. 前，中，后序遍历
# 3. 找二叉树的叶子节点
class BinaryTreeNode(object):
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

class BinaryTree(object):
    def __init__(self, root=None):
        self.root = root
        self.leaf = []  #  存放叶节点


    def is_empty(self):
        return self.root == None

    def preOrder(self,BinaryTreeNode):
        if BinaryTreeNode == None:
            return
        # 先打印根结点，再打印左结点，后打印右结点
        print(BinaryTreeNode.data)
        self.preOrder(BinaryTreeNode.left)
        self.preOrder(BinaryTreeNode.right)

    def inOrder(self,BinaryTreeNode):
        if BinaryTreeNode == None:
            return
        # 先打印左结点，再打印根结点，后打印右结点
        self.inOrder(BinaryTreeNode.left)
        print(BinaryTreeNode.data)
        self.inOrder(BinaryTreeNode.right)

    def postOrder(self,BinaryTreeNode):
        if BinaryTreeNode == None:
            return
        # 先打印左结点，再打印右结点，后打印根结点
        self.postOrder(BinaryTreeNode.left)
        self.postOrder(BinaryTreeNode.right)
        print(BinaryTreeNode.data)
    def getLeaf(self, BinaryTreeNode):
        if BinaryTreeNode and not BinaryTreeNode.left and not BinaryTreeNode.right:
            self.leaf.append(BinaryTreeNode.data)
        if not BinaryTreeNode:
            return
        self.getLeaf(BinaryTreeNode.left)
        self.getLeaf(BinaryTreeNode.right)

n1 = BinaryTreeNode(data="D")
n2 = BinaryTreeNode(data="E")
n3 = BinaryTreeNode(data="F")  #  三个叶子节点
n4 = BinaryTreeNode(data="B", left=n1, right=n2)
n5 = BinaryTreeNode(data="C", left=n3, right=None)
root = BinaryTreeNode(data="A", left=n4, right=n5)

bt = BinaryTree(root)

print("leaf node is:")
bt.getLeaf(bt.root)
print(bt.leaf)
