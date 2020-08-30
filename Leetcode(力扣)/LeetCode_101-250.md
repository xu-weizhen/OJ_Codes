[toc]



# [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

难度 中等

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。 （即逐层地，从左到右访问所有节点）。

 

**示例：**
二叉树：`[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```



**解法**

记录当前层的节点个数或使用两个列表保存该层和下一层的节点。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root == None:
            return []

        layer1 = [root]
        layer2 = []
        ans = []

        while layer1 != []:
            ans.append([])
            for node in layer1:
                ans[-1].append(node.val)

                if node.left:
                    layer2.append(node.left)
                
                if node.right:
                    layer2.append(node.right)
            
            layer1 = layer2
            layer2 = []

        return ans 
```



# [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

难度 简单

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明: **叶子节点是指没有子节点的节点。

**示例：**

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。


**解法**

递归。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。


**代码**

```python
# python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:

        def getDepth(root, depth=1):

            if root == None:
                return depth - 1
            
            left = getDepth(root.left, depth + 1)
            right = getDepth(root.right, depth + 1)
            return max(left, right)

        return getDepth(root)
```

```cpp
// c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(root == nullptr)
            return 0;

        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
        
    }
};
```

```javascript
// javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number}
 */
var maxDepth = function(root) {
    if(root == null)
        return 0;
    
    return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
};
```

```java
// java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null)
            return 0;
        
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```



# [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

难度 中等

根据一棵树的前序遍历与中序遍历构造二叉树。

**注意:**
你可以假设树中没有重复的元素。

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```

返回如下的二叉树：

```
    3
   / \
  9  20
    /  \
   15   7
```



**解法**

递归。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# 官方题解
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:

        def myBuildTree(preorder_left: int, preorder_right: int, inorder_left: int, inorder_right: int):
            if preorder_left > preorder_right:
                return None

            preorder_root = preorder_left

            # 在中序遍历中定位根节点
            inorder_root = index[preorder[preorder_root]]

            root = TreeNode(preorder[preorder_root])

            # 得到左子树中的节点数目
            size_left_subtree = inorder_root - inorder_left

            # 递归地构造左子树
            root.left = myBuildTree(preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1)

            # 递归地构造右子树
            root.right = myBuildTree(preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right)
            
            return root
        
        n = len(preorder)
        # 构造哈希映射，帮助我们快速定位根节点
        index = {element: i for i, element in enumerate(inorder)}
        return myBuildTree(0, n - 1, 0, n - 1)
```



# [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

难度 中等

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1。

**示例:**

```
给定的有序链表： [-10, -3, 0, 5, 9],

一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5
```



**解法**

分治。链表本身为中序遍历结果。将中位点作为根节点，递归建立左右子节点。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:

        def getLength(head):
            ret = 0
            while head:
                ret += 1
                head = head.next
            return ret 
        
        def build(left:int, right:int):
            if left > right:
                return None
            
            mid = (left + right) // 2
            root = TreeNode()
            root.left = build(left, mid - 1)
            nonlocal head
            root.val = head.val
            head = head.next
            root.right = build(mid + 1, right)
            return root

        length = getLength(head)
        return build(0, length - 1)
```



# [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

难度简单422

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

> 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过1。

**示例 1:**

给定二叉树 `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

返回 `true` 。

**示例 2:**

给定二叉树 `[1,2,2,3,3,null,null,4,4]`

```
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

返回 `false` 。



**解法**

递归。自底向上判断每个节点是否符合要求。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        result = self._isBalanced(root, 0)
        return False if result == -1 else True


    def _isBalanced(self, node, depth):
        if node is None:
            return depth
        
        left = self._isBalanced(node.left, depth)
        right = self._isBalanced(node.right, depth)

        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        else:
            return max(left, right) + 1
```



# [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

难度 简单

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明:** 叶子节点是指没有子节点的节点。

**示例:**

给定二叉树 `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最小深度  2.



**解法**

广度优先搜索。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        que = deque()
        que.append((root, 1))

        while que:
            node, deep = que.popleft()
            if node.left is None and node.right is None:
                break
            if node.left:
                que.append((node.left, deep + 1))
            if node.right:
                que.append((node.right, deep + 1))
        
        return deep
```



# [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

难度 简单

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

**说明:** 叶子节点是指没有子节点的节点。

**示例:** 
给定如下二叉树，以及目标和 `sum = 22`，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
```

返回 `true`, 因为存在目标和为 22 的根节点到叶子节点的路径 `5->4->11->2`。



**解法**

广度优先搜索。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import queue

class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
            
        q = queue.Queue()
        q.put([root, root.val])
        ans = False

        while not q.empty():
            node = q.get()
            if node[1] == sum and node[0].left == None and node[0].right == None:
                ans = True
                break
            else:
                if node[0].left != None:
                    q.put([node[0].left, node[1] + node[0].left.val])

                if node[0].right != None:
                    q.put([node[0].right, node[1] + node[0].right.val])
        
        return ans 
```



# [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

难度 中等

给定一个二叉树，[原地](https://baike.baidu.com/item/原地算法/8010757)将它展开为一个单链表。

 

例如，给定二叉树

```
    1
   / \
  2   5
 / \   \
3   4   6
```

将其展开为：

```
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```



**解法**

在先序遍历的同时，记住前一个节点，在遍历的同时构建结果。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        
        stack = [root]
        prev = None
        
        while stack:
            curr = stack.pop()
            if prev:
                prev.left = None
                prev.right = curr
            left, right = curr.left, curr.right
            if right:
                stack.append(right)
            if left:
                stack.append(left)
            prev = curr
```



# [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

难度 中等

给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

**相邻的结点** 在这里指的是 `下标` 与 `上一层结点下标` 相同或者等于 `上一层结点下标 + 1` 的两个结点。

 

例如，给定三角形：

```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

自顶向下的最小路径和为 `11`（即，**2** + **3** + **5** + **1** = 11）。

 

**说明：**

如果你可以只使用 *O*(*n*) 的额外空间（*n* 为三角形的总行数）来解决这个问题，那么你的算法会很加分。



**解法**

将金字塔左对齐，动态规划。时间复杂度： $O(N^2)$ ，空间复杂度： $O(N)$ ， $N$ 为三角形行数



**代码**

```python
# 官方题解
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        f = [0] * n
        f[0] = triangle[0][0]

        for i in range(1, n):
            f[i] = f[i - 1] + triangle[i][i]
            for j in range(i - 1, 0, -1):
                f[j] = min(f[j - 1], f[j]) + triangle[i][j]
            f[0] += triangle[i][0]
        
        return min(f)
```



# [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

难度 简单

给定一个数组，它的第 *i* 个元素是一支给定股票第 *i* 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。

注意你不能在买入股票前卖出股票。

**示例 1:**

```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

**示例 2:**

```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```



**解法**

保存到当前为止价格的最小值和可获得的最大利润，用当前价格减去历史最小值，比较当前利润是否大于历史最大利润。时间复杂度：$O(n)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = int(1e9)
        profit = 0
        for price in prices:
            profit = max(price - min_price, profit)
            min_price = min(price, min_price)
        return profit
```



# [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

难度 困难

给定一个**非空**二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径**至少包含一个**节点，且不一定经过根节点。

**示例 1:**

```
输入: [1,2,3]

       1
      / \
     2   3

输出: 6
```

**示例 2:**

```
输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42
```



**解法**

递归计算左右子节点的贡献值，并更新最大路径值。时间复杂度： $O(N)$ ，空间复杂度 $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.maxSum = float("-inf")

    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            if not node:
                return 0

            # 递归计算左右子节点的最大贡献值
            # 只有在最大贡献值大于 0 时，才会选取对应子节点
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            
            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            priceNewpath = node.val + leftGain + rightGain
            
            # 更新答案
            self.maxSum = max(self.maxSum, priceNewpath)
        
            # 返回节点的最大贡献值
            return node.val + max(leftGain, rightGain)
   
        maxGain(root)
        return self.maxSum
```



# [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

难度 简单

给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

**说明：**本题中，我们将空字符串定义为有效的回文串。

**示例 1:**

```
输入: "A man, a plan, a canal: Panama"
输出: true
```

**示例 2:**

```
输入: "race a car"
输出: false
```



**解法**

+ 方法一：双指针遍历。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。
+ 方法二：向转换字符串，再进行判断。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# 方法一
class Solution:
    def isPalindrome(self, s: str) -> bool:
        if not s:
            return True
        
        left = 0
        right = len(s) - 1
        
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            
            # if left > right or not (s[left] == s[right] or s[left].isalpha() and s[right].isalpha() and abs(ord(s[left]) - ord(s[right])) == 32):
            if left < right and s[left].lower() != s[right].lower():
                break
            
            left += 1
            right -= 1
        
        if left < right:
            return False
        else:
            return True
        
# 方法二
class Solution:
    def isPalindrome(self, s: str) -> bool:
        st = "".join(ch.lower() for ch in s if ch.isalnum())
        return st == st[::-1]
```



# [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

难度 困难

给定两个单词（*beginWord* 和 *endWord*）和一个字典 *wordList*，找出所有从 *beginWord* 到 *endWord* 的最短转换序列。转换需遵循如下规则：

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

- 如果不存在这样的转换序列，返回一个空列表。
- 所有单词具有相同的长度。
- 所有单词只由小写字母组成。
- 字典中不存在重复的单词。
- 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

**示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
```

**示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

输出: []

解释: endWord "cog" 不在字典中，所以不存在符合要求的转换序列。
```



**解法**

构造无向表，进行广度优先搜索。时间复杂度： $O(N^2C)$  ，空间复杂度： $O(N^2)$ ，其中 N 为 `wordList` 的长度，C 为列表中单词的长度。 



**代码**

```java
// 官方题解
class Solution {
    private static final int INF = 1 << 20;
    private Map<String, Integer> wordId; // 单词到id的映射
    private ArrayList<String> idWord; // id到单词的映射
    private ArrayList<Integer>[] edges; // 图的边

    public Solution() {
        wordId = new HashMap<>();
        idWord = new ArrayList<>();
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        int id = 0;
        // 将wordList所有单词加入wordId中 相同的只保留一个 // 并为每一个单词分配一个id
        for (String word : wordList) {
            if (!wordId.containsKey(word)) { 
                wordId.put(word, id++);
                idWord.add(word);
            }
        }
        // 若endWord不在wordList中 则无解
        if (!wordId.containsKey(endWord)) {
            return new ArrayList<>();
        }
        // 把beginWord也加入wordId中
        if (!wordId.containsKey(beginWord)) {
            wordId.put(beginWord, id++);
            idWord.add(beginWord);
        }

        // 初始化存边用的数组
        edges = new ArrayList[idWord.size()];
        for (int i = 0; i < idWord.size(); i++) {
            edges[i] = new ArrayList<>();
        }
        // 添加边
        for (int i = 0; i < idWord.size(); i++) {
            for (int j = i + 1; j < idWord.size(); j++) {
                // 若两者可以通过转换得到 则在它们间建一条无向边
                if (transformCheck(idWord.get(i), idWord.get(j))) {
                    edges[i].add(j);
                    edges[j].add(i);
                }
            }
        }

        int dest = wordId.get(endWord); // 目的ID
        List<List<String>> res = new ArrayList<>(); // 存答案
        int[] cost = new int[id]; // 到每个点的代价
        for (int i = 0; i < id; i++) {
            cost[i] = INF; // 每个点的代价初始化为无穷大
        }

        // 将起点加入队列 并将其cost设为0
        Queue<ArrayList<Integer>> q = new LinkedList<>();
        ArrayList<Integer> tmpBegin = new ArrayList<>();
        tmpBegin.add(wordId.get(beginWord));
        q.add(tmpBegin);
        cost[wordId.get(beginWord)] = 0;

        // 开始广度优先搜索
        while (!q.isEmpty()) {
            ArrayList<Integer> now = q.poll();
            int last = now.get(now.size() - 1); // 最近访问的点
            if (last == dest) { // 若该点为终点则将其存入答案res中
                ArrayList<String> tmp = new ArrayList<>();
                for (int index : now) {
                    tmp.add(idWord.get(index)); // 转换为对应的word
                }
                res.add(tmp);
            } else { // 该点不为终点 继续搜索
                for (int i = 0; i < edges[last].size(); i++) {
                    int to = edges[last].get(i);
                    // 此处<=目的在于把代价相同的不同路径全部保留下来
                    if (cost[last] + 1 <= cost[to]) {
                        cost[to] = cost[last] + 1;
                        // 把to加入路径中
                        ArrayList<Integer> tmp = new ArrayList<>(now); tmp.add(to);
                        q.add(tmp); // 把这个路径加入队列
                    }
                }
            }
        }
        return res;
    }

    // 两个字符串是否可以通过改变一个字母后相等
    boolean transformCheck(String str1, String str2) {
        int differences = 0;
        for (int i = 0; i < str1.length() && differences < 2; i++) {
            if (str1.charAt(i) != str2.charAt(i)) {
                ++differences;
            }
        }
        return differences == 1;
    } 
}
```



# [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

难度 困难

给定一个未排序的整数数组，找出最长连续序列的长度。

要求算法的时间复杂度为 *O(n)*。

**示例:**

```
输入: [100, 4, 200, 1, 3, 2]
输出: 4
解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
```



**解法**

使用集合去重并成为哈希表，对于集合中的每个数，如果该数减一的值不在表中，则从该数开始查找连续序列长度。时间复杂度： $O(N)$， 空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest = 0
        nums = set(nums)

        for num in nums:
            if num - 1 not in nums:
                current = 1
                n = num

                while n + 1 in nums:
                    current += 1
                    n += 1
                
                longest = max(longest, current)
        
        return longest
```



# [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

难度中等309

给定一个二维的矩阵，包含 `'X'` 和 `'O'`（**字母 O**）。

找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

**示例:**

```
X X X X
X O O X
X X O X
X O X X
```

运行你的函数后，矩阵变为：

```
X X X X
X X X X
X X X X
X O X X
```

**解释:**

被围绕的区间不会存在于边界上，换句话说，任何边界上的 `'O'` 都不会被填充为 `'X'`。 任何不在边界上，或不与边界上的 `'O'` 相连的 `'O'` 最终都会被填充为 `'X'`。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。



**解法**

将边界的 `O` 入队，广度优先搜索与边界 `O` 相连的区域，并进行标记。最后将没有标记到的 `O` 区域即为所要找的区域。时间复杂度： $O(nm)$ ，空间复杂度： $O(nm)$ 。



**代码**

```python
# 广度优先
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if board == []:
            return 

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        que = collections.deque()

        for i in range(len(board)):
            if board[i][0] == 'O':
                que.append((i, 0))
            if board[i][-1] == 'O':
                que.append((i, len(board[0]) - 1))
        for i in range(len(board[0])):
            if board[0][i] == 'O':
                que.append((0, i))
            if board[-1][i] == 'O':
                que.append((len(board) - 1, i))
        
        while que:
            x, y = que.popleft()
            board[x][y] = 'n'

            for i in range(4):
                xx = x + dx[i]
                yy = y + dy[i]
                # for xx, yy in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                if xx >= 0 and xx < len(board) and yy >= 0 and yy < len(board[0]) and board[xx][yy] == 'O':
                    que.append((xx, yy))
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'n':
                    board[i][j] = 'O'
                    
        
# 深度优先
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if board == []:
            return 

        def dfs(x, y):
            board[x][y] = 'n'

            for xx, yy in [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]:
                if xx >= 0 and xx < len(board) and yy >= 0 and yy < len(board[0]) and board[xx][yy] == 'O':
                    dfs(xx, yy)

        for i in range(len(board)):
            if board[i][0] == 'O':
                dfs(i, 0)
            if board[i][-1] == 'O':
                dfs(i, len(board[0]) - 1)
        for i in range(len(board[0])):
            if board[0][i] == 'O':
                dfs(0, i)
            if board[-1][i] == 'O':
                dfs(len(board) - 1, i)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'n':
                    board[i][j] = 'O'
```



# [133. 克隆图](https://leetcode-cn.com/problems/clone-graph/)

难度 中等

给你无向 **[连通](https://baike.baidu.com/item/连通图/6460995?fr=aladdin)** 图中一个节点的引用，请你返回该图的 [**深拷贝**](https://baike.baidu.com/item/深拷贝/22785317?fr=aladdin)（克隆）。

图中的每个节点都包含它的值 `val`（`int`） 和其邻居的列表（`list[Node]`）。

```
class Node {
    public int val;
    public List<Node> neighbors;
}
```

 

**测试用例格式：**

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（`val = 1`），第二个节点值为 2（`val = 2`），以此类推。该图在测试用例中使用邻接列表表示。

**邻接列表** 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 **给定节点的拷贝** 作为对克隆图的引用返回。

 

**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/133_clone_graph_question.png)

```
输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/graph.png)

```
输入：adjList = [[]]
输出：[[]]
解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。
```

**示例 3：**

```
输入：adjList = []
输出：[]
解释：这个图是空的，它不含任何节点。
```

**示例 4：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/02/01/graph-1.png)

```
输入：adjList = [[2],[1]]
输出：[[2],[1]]
```

 

**提示：**

1. 节点数不超过 100 。
2. 每个节点值 `Node.val` 都是唯一的，`1 <= Node.val <= 100`。
3. 无向图是一个[简单图](https://baike.baidu.com/item/简单图/1680528?fr=aladdin)，这意味着图中没有重复的边，也没有自环。
4. 由于图是无向的，如果节点 *p* 是节点 *q* 的邻居，那么节点 *q* 也必须是节点 *p* 的邻居。
5. 图是连通图，你可以从给定节点访问到所有节点。



**解法**

广度优先搜索或深度优先搜索。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None

        visited = {}
        que = collections.deque()
        que.append(node)
        visited[node] = Node(node.val, [])

        while que:
            n = que.popleft()
            
            for neighbor in n.neighbors:
                if neighbor not in visited:
                    visited[neighbor] = Node(neighbor.val, [])
                    que.append(neighbor)
                visited[n].neighbors.append(visited[neighbor])

        return visited[node]
```



# [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

难度 简单

给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

**示例 1:**

```
输入: [2,2,1]
输出: 1
```

**示例 2:**

```
输入: [4,1,2,1,2]
输出: 4
```



**思路**

异或位运算。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num
        
        return ans
```



# [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

难度 中等

给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```



**解法**

动态规划。时间复杂度： $O(N^2)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dic = {}
        for word in wordDict:
            dic[word] = 0

        dp = [False] * (len(s) + 1) 
        dp[0] = True 
        for i in range(1, len(s) + 1):
            for j in range(0, i):
                if dp[j] and s[j : i] in dic:
                    dp[i] = True 
                    break 
        
        return dp[-1]
```



# [146. LRU缓存机制](https://leetcode-cn.com/problems/lru-cache/)

难度 中等 

运用你所掌握的数据结构，设计和实现一个 [LRU (最近最少使用) 缓存机制](https://baike.baidu.com/item/LRU)。它应该支持以下操作： 获取数据 `get` 和 写入数据 `put` 。

获取数据 `get(key)` - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
写入数据 `put(key, value)` - 如果密钥已经存在，则变更其数据值；如果密钥不存在，则插入该组「密钥/数据值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

 

**进阶:**

你是否可以在 **O(1)** 时间复杂度内完成这两种操作？

 

**示例:**

```
LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回  1
cache.put(3, 3);    // 该操作会使得密钥 2 作废
cache.get(2);       // 返回 -1 (未找到)
cache.put(4, 4);    // 该操作会使得密钥 1 作废
cache.get(1);       // 返回 -1 (未找到)
cache.get(3);       // 返回  3
cache.get(4);       // 返回  4
```



**解法**

哈希表+双向队列。时间复杂度： $O(1)$ ，空间复杂度： $O(capacity)$ 。



**代码**

```python
class Node:
    def __init__(self, key = 0, value = 0):
        self.pre = None
        self.nex = None
        self.key = key
        self.val = value


class LRUCache:

    def __init__(self, capacity: int):
        self.m = {}

        self.head = Node()
        self.tail = Node()
        self.head.nex = self.tail
        self.tail.pre = self.head

        self.capacity = capacity
        self.used = 0


    def get(self, key: int) -> int:
        if key not in self.m:
            return -1
        
        self.moveToHead(self.m[key])
        return self.m[key].val


    def put(self, key: int, value: int) -> None:
        if key in self.m:
            self.m[key].val = value
            self.moveToHead(self.m[key])
        else:
            if self.used == self.capacity:
                del self.m[self.tail.pre.key]
                self.delTail()
                self.used -= 1

            node = Node(key, value)
            self.m[key] = node
            self.addToHead(node)
            self.used += 1

        # print(self.m)
        # t = self.head
        # while t != None:
        #     print(t.key, end=' ')
        #     t = t.nex
    
    def moveToHead(self, node):
        node.pre.nex = node.nex
        node.nex.pre = node.pre
        self.addToHead(node)
    
    def addToHead(self, node):
        node.nex = self.head.nex
        node.pre = self.head
        node.pre.nex = node
        node.nex.pre = node
    
    def delTail(self):
        self.tail.pre.pre.nex = self.tail
        self.tail.pre = self.tail.pre.pre


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



# [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

难度 中等

给定一个字符串，逐个翻转字符串中的每个单词。

 

**示例 1：**

```
输入: "the sky is blue"
输出: "blue is sky the"
```

**示例 2：**

```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```

**示例 3：**

```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

 

**说明：**

- 无空格字符构成一个单词。
- 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
- 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

 

**进阶：**

请选用 C 语言的用户尝试使用 *O*(1) 额外空间复杂度的原地解法。



**解法**

+ 方法一：使用 `python` 进行切片和翻转，然后进行拼接。时间复杂度：$O(N)$ ，空间复杂度：$O(N)$ 。
+ 方法二：将字符串整体翻转，然后再对每个字符进行翻转。时间复杂度：$O(n)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(reversed(s.split()))
```

```cpp
// 方法二
class Solution {
public:
    string reverseWords(string s) {
        // 反转整个字符串
        reverse(s.begin(), s.end());

        int n = s.size();
        int idx = 0;
        for (int start = 0; start < n; ++start) {
            if (s[start] != ' ') {
                // 填一个空白字符然后将idx移动到下一个单词的开头位置
                if (idx != 0) s[idx++] = ' ';

                // 循环遍历至单词的末尾
                int end = start;
                while (end < n && s[end] != ' ') s[idx++] = s[end++];

                // 反转整个单词
                reverse(s.begin() + idx - (end - start), s.begin() + idx);

                // 更新start，去找下一个单词
                start = end;
            }
        }
        s.erase(s.begin() + idx, s.end());
        return s;
    }
};
```



# [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

难度 中等

给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

 

**示例 1:**

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

**示例 2:**

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```



**思路**

动态规划。当前数为正数时，希望之前的乘积尽可能大；当前数为负数时，希望之前的乘积尽可能小。由此得到转移方程：
$$
f_{max}(i) = \max_{i=1}^{n}\{f_{max}(i-1)*a_i, f_{min}(i-1)*a_i, a_i\} \\
f_{min}(i) = \min_{i=1}^{n}\{f_{max}(i-1)*a_i, f_{min}(i-1)*a_i, a_i\}
$$
时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```cpp
// 官方题解
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        for (int i = 1; i < nums.size(); ++i) {
            int mx = maxF, mn = minF;
            maxF = max(mx * nums[i], max(nums[i], mn * nums[i]));
            minF = min(mn * nums[i], min(nums[i], mx * nums[i]));
            ans = max(maxF, ans);
        }
        return ans;
    }
};
```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if nums == []:
            return 0

        maxF = nums[0]
        minF = nums[0]
        ans = nums[0]

        for i in range(1, len(nums)):
            mmax = maxF
            mmin = minF

            maxF = max(mmax * nums[i], nums[i], mmin * nums[i])
            minF = min(mmin * nums[i], nums[i], mmax * nums[i])

            ans = max(maxF, ans)
        
        return ans
```





# [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

难度 简单

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

- `push(x)` —— 将元素 x 推入栈中。
- `pop()` —— 删除栈顶的元素。
- `top()` —— 获取栈顶元素。
- `getMin()` —— 检索栈中的最小元素。

 

**示例:**

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

 

**提示：**

- `pop`、`top` 和 `getMin` 操作总是在 **非空栈** 上调用。



**解法**

使用辅助栈。时间复杂度： $O(1)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if self.minStack == [] or self.minStack[-1] >= x:
            self.minStack.append(x)

    def pop(self) -> None:
        if self.stack != []:
            if self.stack[-1] == self.minStack[-1]:
                del self.minStack[-1]
            
            del self.stack[-1]

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```



# [167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

难度 简单

给定一个已按照***升序排列\*** 的有序数组，找到两个数使得它们相加之和等于目标数。

函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2*。*

**说明:**

- 返回的下标值（index1 和 index2）不是从零开始的。
- 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

**示例:**

```
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```



**解法**

双指针。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 



**代码**

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        index1 = 0
        index2 = len(numbers) - 1

        while index1 < index2:
            if numbers[index1] + numbers[index2] == target:
                break
            elif numbers[index1] + numbers[index2] < target:
                index1 += 1
            else:
                index2 -= 1
        
        return [index1 + 1, index2 + 1]
        
```



# [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

难度 简单

给定一个大小为 *n* 的数组，找到其中的多数元素。多数元素是指在数组中出现次数**大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1:**

```
输入: [3,2,3]
输出: 3
```

**示例 2:**

```
输入: [2,2,1,1,1,2,2]
输出: 2
```



**解法**

记录下一个数字（初始为数组的第一个元素）和它出现的次数（初始为1），遍历数组，如果遇到的数字和当前记录的数字相同，则次数加一，如果不同，则次数减一。如果次数为0，则修改数字为当前元素的值。因为目标值出现的次数超过数组长度的一半，所以最后剩下的一定为目标值。时间复杂度：$O(n)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        result = nums[0]
        count = 1
        for i in range(1, len(nums)):
            if result != nums[i]:
                if count > 1:
                    count -= 1
                else:
                    result = nums[i]
                    count = 1
            else:
                count += 1
        return result 
```



# [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

难度 困难

一些恶魔抓住了公主（**P**）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（**K**）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。

骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。

有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为*负整数*，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 *0*），要么包含增加骑士健康点数的魔法球（若房间里的值为*正整数*，则表示骑士将增加健康点数）。

为了尽快到达公主，骑士决定每次只向右或向下移动一步。

 

**编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。**

例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 `右 -> 右 -> 下 -> 下`，则骑士的初始健康点数至少为 **7**。

| -2 (K) | -3   | 3      |
| ------ | ---- | ------ |
| -5     | -10  | 1      |
| 10     | 30   | -5 (P) |

 

**说明:**

- 骑士的健康点数没有上限。
- 任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。



**解法**

动态规划，从右向左计算。时间复杂度： $O(MN)$ ，空间复杂度： $O(MN)$ 。



**代码**

```python
# 官方题解
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        n, m = len(dungeon), len(dungeon[0])
        BIG = 10**9
        dp = [[BIG] * (m + 1) for _ in range(n + 1)]
        dp[n][m - 1] = dp[n - 1][m] = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                minn = min(dp[i + 1][j], dp[i][j + 1])
                dp[i][j] = max(minn - dungeon[i][j], 1)

        return dp[0][0]
```



# [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

难度 简单

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**示例 1:**

```
输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

**示例 2:**

```
输入: [2,7,9,3,1]
输出: 12
解释: 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```



**解法**

动态规划。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        if len(nums) == 0:
            return 0
        elif len(nums) == 1:
            return nums[0]

        size = len(nums)
        dp = [0] * size
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, size):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        
        return dp[size - 1]
```



# [199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

难度 中等

给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**示例:**

```
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```



**解法**

将每一层的节点从左到右放入数组，则数组最后一个节点为所看到的节点。根据数组对下一层进行同样操作。时间复杂度：$O(N)$ ，空间复杂度：$O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
            
        res = [root.val]
        layers = [root]

        while layers:
            l = []
            for node in layers:
                if node.left:
                    l.append(node.left)
                if node.right:
                    l.append(node.right)
            if l:
                res.append(l[-1].val)
            layers = l 
        
        return res
```



# [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

难度 中等

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例 1:**

```
输入:
11110
11010
11000
00000
输出: 1
```

**示例 2:**

```
输入:
11000
11000
00100
00011
输出: 3
解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。
```



**解法**

每个点进行深度优先搜索。时间复杂度：$O(MN)$ ，空间复杂度：$O(MN)$ 。 $M,N$ 是地图长度和宽度。



**代码**

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        if m == 0:
            return 0

        n = len(grid[0])

        find = [[True for i in range(n)] for j in range(m)]
        ans = 0

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        def DFS(i, j):
            find[i][j] = False
            for k in range(4):
                x = i + dx[k]
                y = j + dy[k]
                if x >= 0 and x < m and y >= 0 and y < n and find[x][y] and grid[x][y] == '1':
                    BFS(x, y)


        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and find[i][j]:
                    DFS(i, j)
                    ans += 1
        
        return ans
```



# [201. 数字范围按位与](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)

难度 中等

给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

**示例 1:** 

```
输入: [5,7]
输出: 4
```

**示例 2:**

```
输入: [0,1]
输出: 0
```



**解法**

[Brian Kernighan 算法](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/solution/shu-zi-fan-wei-an-wei-yu-by-leetcode-solution/) ，每次对 `number` 和 `number-1` 之间进行按位与运算后，`number` 中最右边的 `1` 会被抹去变成 `0` 。基于上述技巧，可以用它来计算两个二进制字符串的公共前缀。时间复杂度： $O(\log n)$ ，空间复杂度： $O(1)$ 。

<img src="https://assets.leetcode-cn.com/solution-static/201/9.png" style="zoom: 40%;" />



**代码**

```python
# 官方题解
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        while m < n:
            # 抹去最右边的 1
            n = n & (n - 1)    
        return n
```







# [202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

难度 简单

编写一个算法来判断一个数 `n` 是不是快乐数。

「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。如果 **可以变为** 1，那么这个数就是快乐数。

如果 `n` 是快乐数就返回 `True` ；不是，则返回 `False` 。

 

**示例：**

```
输入：19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```



**解法**

使用哈希表保存已经遇到过的数字，对数组按要求进行操作，如果出现哈希表中出现过的数字，则陷入循环，不是快乐数。时间复杂度： $O(\log N)$ ，空间复杂度： $O(\log N)$ 。



**代码**

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        d ={}	# 也可用set()

        while n not in d and n != 1:
            d[n] = 1
            s = 0
            while n > 0:
                s += (n % 10) ** 2
                n = n // 10
            n = s
        
        if n == 1:
            return True
        else:
            return False
```



# [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

难度 简单

反转一个单链表。

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```



**代码**

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next)
            return head;

        ListNode* prev = NULL;
        ListNode* next = head->next;

        while(next)
        {
            head->next = prev;
            prev = head;
            head = next;
            next = next->next;
        }

        head->next = prev;
        return head;
    }
};
```

``` python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        last = None
        now = head

        while now:
            now.next, last, now = last, now, now.next

        return last
```

``` javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var reverseList = function(head) {
    var prev = null;
    var now = head;

    while(now){
        [now.next, prev, now] = [prev, now, now.next];
    }

    return prev;
};
```



# [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

难度 中等

你这个学期必须选修 `numCourse` 门课程，记为 `0` 到 `numCourse-1` 。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们：`[0,1]`

给定课程总量以及它们的先决条件，请你判断是否可能完成所有课程的学习？

 

**示例 1:**

```
输入: 2, [[1,0]] 
输出: true
解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以这是可能的。
```

**示例 2:**

```
输入: 2, [[1,0],[0,1]]
输出: false
解释: 总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0；并且学习课程 0 之前，你还应先完成课程 1。这是不可能的。
```

 

**提示：**

1. 输入的先决条件是由 **边缘列表** 表示的图形，而不是 邻接矩阵 。详情请参见[图的表示法](http://blog.csdn.net/woaidapaopao/article/details/51732947)。
2. 你可以假定输入的先决条件中没有重复的边。
3. `1 <= numCourses <= 10^5`



**解法**

记录每个节点的入度，入度为 0 时移除该节点，若最终没有剩余节点，则可以完成学习。时间复杂度： $O(len(prerequisites)^2)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        edges = collections.defaultdict(list)
        indeg = [0] * numCourses

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1

        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
        visited = 0

        while q:
            visited += 1
            u = q.popleft()
            for e in edges[u]:
                indeg[e] -= 1
                if indeg[e] == 0:
                    q.append(e)
                
        return visited == numCourses
```



# [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

难度 中等

给定一个含有 **n** 个正整数的数组和一个正整数 **s ，**找出该数组中满足其和 **≥ s** 的长度最小的连续子数组，并返回其长度**。**如果不存在符合条件的连续子数组，返回 0。

**示例:** 

```
输入: s = 7, nums = [2,3,1,2,4,3]
输出: 2
解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。
```

**进阶:**

如果你已经完成了*O*(*n*) 时间复杂度的解法, 请尝试 *O*(*n* log *n*) 时间复杂度的解法。



**解法**

双指针。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 



**代码** 

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0

        count = nums[0]
        left = 0
        right = 0
        ans = len(nums)
        find = False

        while True:
            if count < s:
                right += 1
                if right >= len(nums):
                    break
                count += nums[right]
            else:
                ans = min(ans , right - left + 1)
                find = True
                count -= nums[left]
                left += 1
        
        return ans if find else 0
```



# [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

难度 中等

现在你总共有 *n* 门课需要选，记为 `0` 到 `n-1`。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: `[0,1]`

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

**示例 1:**

```
输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
```

**示例 2:**

```
输入: 4, [[1,0],[2,0],[3,1],[3,2]]
输出: [0,1,2,3] or [0,2,1,3]
解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
     因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
```

**说明:**

1. 输入的先决条件是由**边缘列表**表示的图形，而不是邻接矩阵。详情请参见[图的表示法](http://blog.csdn.net/woaidapaopao/article/details/51732947)。
2. 你可以假定输入的先决条件中没有重复的边。

**提示:**

1. 这个问题相当于查找一个循环是否存在于有向图中。如果存在循环，则不存在拓扑排序，因此不可能选取所有课程进行学习。

2. [通过 DFS 进行拓扑排序](https://www.coursera.org/specializations/algorithms) - 一个关于Coursera的精彩视频教程（21分钟），介绍拓扑排序的基本概念。

3. 拓扑排序也可以通过 [BFS](https://baike.baidu.com/item/宽度优先搜索/5224802?fr=aladdin&fromid=2148012&fromtitle=广度优先搜索) 完成。




**解法**

构建有向图，使用广度优先搜索。时间复杂度： $O(M+N)$ ，空间复杂度： $O(M+N)$ 。



**代码**

```python
# 官方题解
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 存储有向图
        edges = collections.defaultdict(list)
        # 存储每个节点的入度
        indeg = [0] * numCourses
        # 存储答案
        result = list()

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indeg[info[0]] += 1
        
        # 将所有入度为 0 的节点放入队列中
        q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])

        while q:
            # 从队首取出一个节点
            u = q.popleft()
            # 放入答案中
            result.append(u)
            for v in edges[u]:
                indeg[v] -= 1
                # 如果相邻节点 v 的入度为 0，就可以选 v 对应的课程了
                if indeg[v] == 0:
                    q.append(v)

        if len(result) != numCourses:
            result = list()
        return result
```



# [214. 最短回文串](https://leetcode-cn.com/problems/shortest-palindrome/)

难度困难255

给定一个字符串 ***s***，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。

**示例 1:**

```
输入: "aacecaaa"
输出: "aaacecaaa"
```

**示例 2:**

```
输入: "abcd"
输出: "dcbabcd"
```



**解法**

KMP算法。时间复杂度：$O(|s|)$ ，空间复杂度： $O(|s|)$ 。



**代码**

```python
# 官方题解
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        n = len(s)
        fail = [-1] * n
        for i in range(1, n):
            j = fail[i - 1]
            while j != -1 and s[j + 1] != s[i]:
                j = fail[j]
            if s[j + 1] == s[i]:
                fail[i] = j + 1
        
        best = -1
        for i in range(n - 1, -1, -1):
            while best != -1 and s[best + 1] != s[i]:
                best = fail[best]
            if s[best + 1] == s[i]:
                best += 1

        add = ("" if best == n - 1 else s[best+1:])
        return add[::-1] + s
```



# [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

难度 中等

在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。

**示例:**

```
输入: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

输出: 4
```



**解法**

动态规划。时间复杂度： $O(MN)$ ，空间复杂度： $O(MN)$ 。



**代码**

```python
# 官方题解
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        
        maxSide = 0
        rows, columns = len(matrix), len(matrix[0])
        dp = [[0] * columns for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    maxSide = max(maxSide, dp[i][j])
        
        maxSquare = maxSide * maxSide
        return maxSquare
```



#  [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

难度 简单

使用队列实现栈的下列操作：

- push(x) -- 元素 x 入栈
- pop() -- 移除栈顶元素
- top() -- 获取栈顶元素
- empty() -- 返回栈是否为空

**注意:**

- 你只能使用队列的基本操作-- 也就是 `push to back`, `peek/pop from front`, `size`, 和 `is empty` 这些操作是合法的。
- 你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
- 你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。



**代码**

``` C++
class MyStack {
public:
    queue<int> q;

    /** Initialize your data structure here. */
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        q.push(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int size = q.size();
        queue<int> qt;
        while(size != 1)
        {
            qt.push(q.front());
            q.pop();
            size = q.size();
        }
        int result  = q.front();
        q.pop();
        while(! qt.empty())
        {
            q.push(qt.front());
            qt.pop();
        }
        return result;
    }
    
    /** Get the top element. */
    int top() {
        return q.back();
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        return q.empty();
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
```

``` python
class MyStack:

    def __init__(self):
        self.q = []

    def push(self, x: int) -> None:
        self.q.append(x)

    def pop(self) -> int:
        result =  self.q[-1]
        del self.q[-1]
        return result

    def top(self) -> int:
        return self.q[-1]

    def empty(self) -> bool:
        return len(self.q) == 0
```



# [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

难度 中等

在未排序的数组中找到第 **k** 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

**示例 1:**

```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

**示例 2:**

```
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```

**说明:**

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。



**代码**

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-k]
```



# [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

难度 中等

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉树: root = [3,5,1,6,2,0,8,null,null,7,4]

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

 

**示例 1:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```

**示例 2:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```

 

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。



**解法**

为原始树构造父节点，保存在哈希表中，查表得到节点的所有父节点。时间复杂度：$O(N)$ ， 空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        tree = {}

        def BuildFather(root):
            if root.left != None:
                tree[root.left.val] = root
                BuildFather(root.left)
            
            if root.right != None:
                tree[root.right.val] = root
                BuildFather(root.right)
            

        BuildFather(root)

        vis = {}

        while True:
            vis[p.val] = True
            if p.val in tree:
                p = tree[p.val]
            else:
                break

        while True:
            if q.val in vis:
                ans = q 
                break
            q = tree[q.val]
        
        return ans
```



# [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

难度 中等

给你一个长度为 *n* 的整数数组 `nums`，其中 *n* > 1，返回输出数组 `output` ，其中 `output[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

 

**示例:**

```
输入: [1,2,3,4]
输出: [24,12,8,6]
```

 

**提示：**题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

**说明:** 请**不要使用除法，**且在 O(*n*) 时间复杂度内完成此题。

**进阶：**
你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组**不被视为**额外空间。）



**解法**

使用数组保存每个数左边的数的乘积和右边的数的乘积，左右乘积相乘即为答案。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ （不计输出）。



**代码**

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if len(nums) <= 1:
            return nums

        # 该数左边数字乘积
        ans = [0] * len(nums)
        ans[0] = 1
        for i in range(1, len(nums)):
            ans[i] = ans[i - 1] * nums[i - 1]

        # 该数右边数字乘积
        r = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            ans[i] = ans[i] * r
            r *= nums[i]

        return ans
```



# [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

难度 困难

给定一个数组 *nums*，有一个大小为 *k* 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 *k* 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

 

**示例:**

```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

 

**提示：**

你可以假设 *k* 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。



**解法**

使用一个双端队列保存当前窗口的递减最大值序列。

- 时间复杂度：$O(N)$ ，空间复杂度： $O(k)$ 



**代码**

``` cpp
#include <deque>
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> q;
        vector<int> res;

        for(int index = 0; index < nums.size(); index++)
        {
            if(index - k >= 0)
            {
                res.push_back(q[0]);
                if(nums[index - k] == q[0])
                    q.pop_front();
            }
            
            while(!q.empty() && q.back() < nums[index])
                q.pop_back();
            q.push_back(nums[index]);
        }

        if(!q.empty())
            res.push_back(q[0]);

        return res;

    }
};
```

``` python
import queue
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = queue.deque()
        res = []

        for index in range(0, len(nums)):
            if index - k >= 0:
                res.append(q[0])
                if nums[index - k] == q[0]:
                    q.popleft()
            
            while q and q[-1] < nums[index]:
                q.pop()
            q.append(nums[index])
        
        if q:
            res.append(q[0])
        return res
```


