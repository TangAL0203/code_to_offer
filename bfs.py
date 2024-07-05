## 关于deque的特点
# 1. 快速从两端添加和删除元素：deque在两端添加和删除元素的时间复杂度都是O(1)
# 2. 线程安全：deque的实例可以在多线程环境中安全使用，而不需要额外的锁定。
# 3. 可选的最大长度：可以通过maxlen参数来限制deque的最大长度。当deque已满时，添加新元素会导致最早添加的元素被自动移除。

# # 创建一个空的deque
# d = deque()

# # 从右侧添加元素
# d.append('a')
# d.append('b')
# print(d)  # 输出：deque(['a', 'b'])

# # 从左侧添加元素
# d.appendleft('c')
# print(d)  # 输出：deque(['c', 'a', 'b'])

# # 从右侧移除元素
# right_item = d.pop()
# print(right_item)  # 输出：'b'
# print(d)  # 输出：deque(['c', 'a'])

# # 从左侧移除元素
# left_item = d.popleft()
# print(left_item)  # 输出：'c'
# print(d)  # 输出：deque(['a'])


from collections import deque
 
def bfs(graph, root):
    visited = set()
    queue = deque()
    
    # 将起始节点加入队列
    queue.append(root)
    visited.add(root)
    
    while len(queue):
        # 获取队列的长度，这样我们就可以一次性遍历当前层的所有节点
        for _ in range(len(queue)):
            current_node = queue.popleft()
            # 处理节点
            print(current_node)
            # 将当前节点的未访问的邻居节点加入队列
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
 
# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}
 
# 从节点'A'开始进行BFS遍历
bfs(graph, 'A')