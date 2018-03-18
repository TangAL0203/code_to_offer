#-*-coding:utf-8-*-
class ListNode(object):  
    def __init__(self):  
        self.val = None  
        self.next = None

def delete_node(root, node):
    # if node.val == root.val and node.next == root.next:
    if node.next == None:
        while root.next.next : # 找到node的上一个节点
            root = root.next
        # 删除node
        root.next = None
    else:
        node.val = node.next.val  # 将node.next值复制给node
        node.next = node.next.next # 将note.next指向node.next.next