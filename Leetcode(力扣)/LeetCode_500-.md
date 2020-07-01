[toc]




# [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

难度 中等

给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

**示例 1:**
输入:

```
0 0 0
0 1 0
0 0 0
```

输出:

```
0 0 0
0 1 0
0 0 0
```

**示例 2:**
输入:

```
0 0 0
0 1 0
1 1 1
```

输出:

```
0 0 0
0 1 0
1 2 1
```

**注意:**

1. 给定矩阵的元素个数不超过 10000。
2. 给定矩阵中至少有一个元素是 0。
3. 矩阵中的元素只在四个方向上相邻: 上、下、左、右。



**解法**

动态规划。先看左和上，再看右和下。时间复杂度：$O(N^2)$，空间复杂度：$O(1)$ ， $N$ 为矩阵宽度。



**代码**

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        m = len(matrix)
        n = len(matrix[0])

        dp = [[10000] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dp[i][j] = 0

        # 左和上
        for i in range(m):
            for j in range(n):
                if i - 1 >= 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
                if j - 1 >= 0: 
                    dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)

        # 右和上
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i + 1 < m:
                    dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
                if j + 1 < n: 
                    dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)

        return dp
```



# [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

难度 简单

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。

**示例 :**
给定二叉树

```
          1
         / \
        2   3
       / \     
      4   5    
```

返回 **3**, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

**注意：**两结点之间的路径长度是以它们之间边的数目表示。



**解法**

遍历当前节点左右子树，获得左右子树的最大深度，与当前最大直径作比较，然后递归对该节点左右子树执行该操作。



**代码**

``` python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.result = 0

        def depth(root):
            if not root:
                return 0
            
            left = depth(root.left)
            right = depth(root.right)
            self.result = max(self.result, left + right)
            # print('root:{}  left:{}  right:{}'.format(root.val, left, right))
            return max(left, right) + 1

        depth(root)
        return self.result
```



# [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

难度 中等

给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

**示例 1 :**

```
输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。
```

**说明 :**

1. 数组的长度为 [1, 20,000]。
2. 数组中元素的范围是 [-1000, 1000] ，且整数 **k** 的范围是 [-1e7, 1e7]。



**解法**

前缀和+哈希表。时间复杂度：$O(N)$，空间复杂度：$O(N)$  。



**代码**

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
        m = {0:1}
        count = 0
        pre = 0

        for num in nums:
            pre += num
            if pre - k in m:
                count += m[pre - k]

            m[pre] = m[pre] + 1 if pre in m else 1
        
        return count
```



# [572. 另一个树的子树](https://leetcode-cn.com/problems/subtree-of-another-tree/)

难度 简单

给定两个非空二叉树 **s** 和 **t**，检验 **s** 中是否包含和 **t** 具有相同结构和节点值的子树。**s** 的一个子树包括 **s** 的一个节点和这个节点的所有子孙。**s** 也可以看做它自身的一棵子树。

**示例 1:**
给定的树 s:

```
     3
    / \
   4   5
  / \
 1   2
```

给定的树 t：

```
   4 
  / \
 1   2
```

返回 **true**，因为 t 与 s 的一个子树拥有相同的结构和节点值。

**示例 2:**
给定的树 s：

```
     3
    / \
   4   5
  / \
 1   2
    /
   0
```

给定的树 t：

```
   4
  / \
 1   2
```

返回 **false**。



**解法**

暴力法。对比 `s` 每个节点的值是否与 `t` 根节点的值一样，若一样则从该节点开始进行比较。时间复杂度： $O(MN)$ ，空间复杂度： $O(1)$ 。 $MN$ 分别为 `s` 和 `t` 的节点数。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:

        def same(sNode, tNode):
            if sNode == None and tNode == None:
                return True

            if sNode.val != tNode.val or \
            		sNode.left != None and tNode.left == None or \
                	sNode.left == None and tNode.left != None or  \
                	sNode.right != None and tNode.right == None or \
                	sNode.right == None and tNode.right != None :
                return False
            
            if same(sNode.left, tNode.left):
                return same(sNode.right, tNode.right)
            else:
                return False
        
        def find(s, t):
            if s.val == t.val:
                if same(s, t):
                    return True

            if s.left != None:
                if find(s.left, t):
                    return True
            
            if s.right != None:
                return find(s.right, t)
            
            return False
        
        return find(s, t)
```



# [680. 验证回文字符串 Ⅱ](https://leetcode-cn.com/problems/valid-palindrome-ii/)

难度 简单

给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

**示例 1:**

```
输入: "aba"
输出: True
```

**示例 2:**

```
输入: "abca"
输出: True
解释: 你可以删除c字符。
```

**注意:**

1. 字符串只包含从 a-z 的小写字母。字符串的最大长度是50000。



**解法**

双指针。时间复杂度：$O(N)$，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        self.l = 0
        self.r = right

        if self.isPalindrome(s, left, right):
            return True
        else:
            left = self.l
            right = self.r
            return self.isPalindrome(s, left + 1, right) or self.isPalindrome(s, left, right - 1)

    def isPalindrome(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                break
            left += 1
            right -= 1
        
        if left >= right:
            return True
        else:
            self.l = left
            self.r = right
            return False
```



# [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

难度 中等

给定一个包含了一些 0 和 1的非空二维数组 `grid` , 一个 **岛屿** 是由四个方向 (水平或垂直) 的 `1` (代表土地) 构成的组合。你可以假设二维矩阵的四个边缘都被水包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0。)

**示例 1:**

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

对于上面这个给定矩阵应返回 `6`。注意答案不应该是11，因为岛屿只能包含水平或垂直的四个方向的‘1’。

**示例 2:**

```
[[0,0,0,0,0,0,0,0]]
```

对于上面这个给定的矩阵, 返回 `0`。

**注意:** 给定的矩阵`grid` 的长度和宽度都不超过 50。



**解法**

广度优先搜索。使用一个数组保存已经访问过的地点，遍历所有地点，对未访问的陆地递归计算其面积，返回最大的面积。时间复杂度：$O(mn)$，空间复杂度：$O(mn)$ 。



**代码**

```python
import copy
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0

        self.row = len(grid)
        self.col = len(grid[0])

        self.record = []
        t = [True] * self.col
        for k in range(self.row):
            self.record.append(copy.copy(t))

        result = 0
        for i in range(self.row):
            for j in range(self.col):
                if grid[i][j] and self.record[i][j]:
                    result = max(result, self.landArea(i, j, grid))
        return result

    def landArea(self, i, j, grid):
        self.record[i][j] = False
        result = 1

        if i - 1 >= 0 and grid[i - 1][j] and self.record[i - 1][j]:
            result += self.landArea(i - 1, j, grid)
        if i + 1 < self.row and grid[i + 1][j] and self.record[i + 1][j]:
            result += self.landArea(i + 1, j, grid)
        if j - 1 >= 0  and grid[i][j - 1] and self.record[i][j - 1]:
            result += self.landArea(i, j - 1, grid)
        if j + 1 < self.col and grid[i][j + 1] and self.record[i][j + 1]:
            result += self.landArea(i, j + 1, grid)
        
        return result
```



# [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

难度 中等

给两个整数数组 `A` 和 `B` ，返回两个数组中公共的、长度最长的子数组的长度。

**示例 1:**

```
输入:
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出: 3
解释: 
长度最长的公共子数组是 [3, 2, 1]。
```

**说明:**

1. 1 <= len(A), len(B) <= 1000
2. 0 <= A[i], B[i] < 100



**解法**

动态规划。时间复杂度： $O(MN)$ ，空间复杂度： $O(MN)$ 。



**代码**

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        n, m = len(A), len(B)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        ans = 0
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                dp[i][j] = dp[i + 1][j + 1] + 1 if A[i] == B[j] else 0
                ans = max(ans, dp[i][j])
        return ans
```



# [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

难度 中等

根据每日 `气温` 列表，请重新生成一个列表，对应位置的输出是需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 `0` 来代替。

例如，给定一个列表 `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`，你的输出应该是 `[1, 1, 4, 2, 1, 1, 0, 0]`。

**提示：**`气温` 列表长度的范围是 `[1, 30000]`。每个气温的值的均为华氏度，都是在 `[30, 100]` 范围内的整数。



**解法**

使用一个栈保存温度，遍历气温列表，当遇到新温度时，如果该温度比栈中的温度高，则更新栈中温度对应日期的答案。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        if not T:
            return []
        
        stack = []
        ans = [0] * len(T)

        for i in range(0, len(T)):
            while stack and T[i] > T[stack[-1]]:
                day = stack.pop()
                ans[day] = i - day 
            
            stack.append(i)
        
        return ans
```



# [820. 单词的压缩编码](https://leetcode-cn.com/problems/short-encoding-of-words/)

难度 中等

给定一个单词列表，我们将这个列表编码成一个索引字符串 `S` 与一个索引列表 `A`。

例如，如果这个列表是 `["time", "me", "bell"]`，我们就可以将其表示为 `S = "time#bell#"` 和 `indexes = [0, 2, 5]`。

对于每一个索引，我们可以通过从字符串 `S` 中索引的位置开始读取字符串，直到 "#" 结束，来恢复我们之前的单词列表。

那么成功对给定单词列表进行编码的最小字符串长度是多少呢？

 

**示例：**

```
输入: words = ["time", "me", "bell"]
输出: 10
说明: S = "time#bell#" ， indexes = [0, 2, 5] 。
```

 

**提示：**

1. `1 <= words.length <= 2000`
2. `1 <= words[i].length <= 7`
3. 每个单词都是小写字母 。



**解法**

将所有单词加入一个列表中，移除每个单词可以产生的后缀。所有单词长度之和，即为结果。

时间复杂度： $O(\sum w_i^2)$ 其中 $w_i$ 是 `words[i]​` 的长度。每个单词有 $w_i$ 个后缀，对于每个后缀，查询其是否在集合中时需要进行 $O(w_i)$ 的哈希值计算；空间复杂度： $O(\sum w_i)$ 。



**代码**

``` python
class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        s = set(words)

        for word in words:
            for i in range(1, len(word)):
                s.discard(word[i:])
        
        res = sum(len(word) + 1 for word in s)
        return res
```



# [836. 矩形重叠](https://leetcode-cn.com/problems/rectangle-overlap/)

难度 简单

矩形以列表 `[x1, y1, x2, y2]` 的形式表示，其中 `(x1, y1)` 为左下角的坐标，`(x2, y2)` 是右上角的坐标。

如果相交的面积为正，则称两矩形重叠。需要明确的是，只在角或边接触的两个矩形不构成重叠。

给出两个矩形，判断它们是否重叠并返回结果。

 

**示例 1：**

```
输入：rec1 = [0,0,2,2], rec2 = [1,1,3,3]
输出：true
```

**示例 2：**

```
输入：rec1 = [0,0,1,1], rec2 = [1,0,2,1]
输出：false
```

 

**提示：**

1. 两个矩形 `rec1` 和 `rec2` 都以含有四个整数的列表的形式给出。
2. 矩形中的所有坐标都处于 `-10^9` 和 `10^9` 之间。
3. `x` 轴默认指向右，`y` 轴默认指向上。
4. 你可以仅考虑矩形是正放的情况。



**解法**

+ 方法一：考虑不存在相交可能的情况，取反。时间复杂度：$O(1)$，空间复杂度：$O(1)$
+ 方法二：投影到 $x$ , $y$ 坐标，检查是否有交集。时间复杂度：$O(1)$，空间复杂度：$O(1)$



**代码**

```python
# 方法二
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        def intersect(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)
        return (intersect(rec1[0], rec1[2], rec2[0], rec2[2]) and
                intersect(rec1[1], rec1[3], rec2[1], rec2[3]))
```



# [837. 新21点](https://leetcode-cn.com/problems/new-21-game/)

难度 中等

爱丽丝参与一个大致基于纸牌游戏 “21点” 规则的游戏，描述如下：

爱丽丝以 `0` 分开始，并在她的得分少于 `K` 分时抽取数字。 抽取时，她从 `[1, W]` 的范围中随机获得一个整数作为分数进行累计，其中 `W` 是整数。 每次抽取都是独立的，其结果具有相同的概率。

当爱丽丝获得不少于 `K` 分时，她就停止抽取数字。 爱丽丝的分数不超过 `N` 的概率是多少？

**示例** **1**：

```
输入：N = 10, K = 1, W = 10
输出：1.00000
说明：爱丽丝得到一张卡，然后停止。
```

**示例** **2**：

```
输入：N = 6, K = 1, W = 10
输出：0.60000
说明：爱丽丝得到一张卡，然后停止。
在 W = 10 的 6 种可能下，她的得分不超过 N = 6 分。
```

**示例** **3**：

```
输入：N = 21, K = 17, W = 10
输出：0.73278
```

**提示：**

1. `0 <= K <= N <= 10000`
2. `1 <= W <= 10000`
3. 如果答案与正确答案的误差不超过 `10^-5`，则该答案将被视为正确答案通过。
4. 此问题的判断限制时间已经减少。



**解法**

+ 方法一：动态规划。 $dp[x]=\frac{dp[x+1]+dp[x+2]+⋯+dp[x+W]}{W}$ 。时间复杂度 $O(N+KW)$ ，空间复杂度：$O(K+W)$ 。
+ 方法二：方法一基础上优化得到  $dp[x]=dp[x+1]−\frac{dp[x+W+1]−dp[x+1]}{W}$ 。时间复杂度 $O(\min(N,K+W)$ ，空间复杂度：$O(K+W)$ 。



**代码**

```python
# 官方题解
# 方法一 (超时)
class Solution:
    def new21Game(self, N: int, K: int, W: int) -> float:
        if K == 0:
            return 1.0
        
        dp = [0.0] * (K + W)
        
        for i in range(K, min(N, K + W - 1) + 1):
            dp[i] = 1.0
            
        for i in range(K - 1, -1, -1):
            for j in range(1, W + 1):
                dp[i] += dp[i + j] / W
                
        return dp[0]


# 方法二
class Solution:
    def new21Game(self, N: int, K: int, W: int) -> float:
        if K == 0:
            return 1.0
        dp = [0.0] * (K + W)
        
        for i in range(K, min(N, K + W - 1) + 1):
            dp[i] = 1.0
            
        dp[K - 1] = float(min(N - K + 1, W)) / W
        
        for i in range(K - 2, -1, -1):
            dp[i] = dp[i + 1] - (dp[i + W + 1] - dp[i + 1]) / W
            
        return dp[0]
```



# [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

难度 简单

给定一个带有头结点 `head` 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

 

**示例 1：**

```
输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
```

**示例 2：**

```
输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
```

 

**提示：**

- 给定链表的结点数介于 `1` 和 `100` 之间。



**解法**

使用快慢两个指针，快指针一次移动两个节点，慢指针一次移动一个节点。快指针到达链表尾时，慢指针达到链表中间。时间复杂度：$O(n)$，空间复杂度：$O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        while fast.next:
            slow = slow.next
            fast = fast.next
            if fast.next:
                fast = fast.next
        
        return slow
```



# [887. 鸡蛋掉落](https://leetcode-cn.com/problems/super-egg-drop/)

难度 困难

你将获得 `K` 个鸡蛋，并可以使用一栋从 `1` 到 `N` 共有 `N` 层楼的建筑。

每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。

你知道存在楼层 `F` ，满足 `0 <= F <= N` 任何从高于 `F` 的楼层落下的鸡蛋都会碎，从 `F` 楼层或比它低的楼层落下的鸡蛋都不会破。

每次*移动*，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 `X` 扔下（满足 `1 <= X <= N`）。

你的目标是**确切地**知道 `F` 的值是多少。

无论 `F` 的初始值如何，你确定 `F` 的值的最小移动次数是多少？

 



**示例 1：**

```
输入：K = 1, N = 2
输出：2
解释：
鸡蛋从 1 楼掉落。如果它碎了，我们肯定知道 F = 0 。
否则，鸡蛋从 2 楼掉落。如果它碎了，我们肯定知道 F = 1 。
如果它没碎，那么我们肯定知道 F = 2 。
因此，在最坏的情况下我们需要移动 2 次以确定 F 是多少。
```

**示例 2：**

```
输入：K = 2, N = 6
输出：3
```

**示例 3：**

```
输入：K = 3, N = 14
输出：4
```

 

**提示：**

1. `1 <= K <= 100`
2. `1 <= N <= 10000`



**解法**

动态规划 + 二分搜索
$$
dp(K,N) = 1 + \min_{1 \le X \le N}(\max{(dp(K−1,X−1),dp(K,N−X))})
$$
在上述的状态转移方程中，第一项 $\mathcal{T_1}(X) = dp(K-1, X-1)$  是一个随 $X$ 的增加而单调递增的函数，第二项 $\mathcal{T_2}(X) = dp(K, N-X)$ 是一个随着 $X$ 的增加而单调递减的函数。使用二分搜索找出这两个函数的交点，在交点处就保 证这两个函数的最大值最小。 

时间复杂度：$O(K * N \log N)$ 。需要计算  个状态，每个状态计算时需要 $O(\log N)$ 的时间进行二分搜索。空间复杂度：$O(K * N)$ 。需要 $O(K * N)$ 的空间存储每个状态的解。



**代码**

```python
class Solution:
    def superEggDrop(self, K: int, N: int) -> int:
        test = {}

        def dp(k, n):
            if (k, n) not in test:
                # 楼层为0
                if n == 0:
                    ans = 0
                # 鸡蛋为1
                elif k == 1:
                    ans = n
                else:
                    # 二分法查找
                    low, high = 1, n
                    while low + 1 < high:
                        x = (low + high) // 2
                        t1 = dp(k-1, x-1)   # 碎了
                        t2 = dp(k, n-x)     # 没碎

                        if t1 < t2:
                            low = x
                        elif t1 > t2:
                            high = x
                        else:
                            low = high = x

                    ans = 1 + min(max(dp(k-1, x-1), dp(k, n-x))
                                  for x in (low, high))

                test[k, n] = ans
            return test[k, n]

        return dp(K, N)
```



# [892. 三维形体的表面积](https://leetcode-cn.com/problems/surface-area-of-3d-shapes/)

难度 简单

在 `N * N` 的网格上，我们放置一些 `1 * 1 * 1 ` 的立方体。

每个值 `v = grid[i][j]` 表示 `v` 个正方体叠放在对应单元格 `(i, j)` 上。

请你返回最终形体的表面积。

 

**示例 1：**

```
输入：[[2]]
输出：10
```

**示例 2：**

```
输入：[[1,2],[3,4]]
输出：34
```

**示例 3：**

```
输入：[[1,0],[0,2]]
输出：16
```

**示例 4：**

```
输入：[[1,1,1],[1,0,1],[1,1,1]]
输出：32
```

**示例 5：**

```
输入：[[2,2,2],[2,1,2],[2,2,2]]
输出：46
```

 

**提示：**

- `1 <= N <= 50`
- `0 <= grid[i][j] <= 50`



**解法**

依次遍历每一个立方体，计算其暴露在外的面积。时间复杂度： $O(n^2)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def surfaceArea(self, grid: List[List[int]]) -> int:
        res = 0
        N = len(grid)

        for i in range(N):
            for j in range(N):
                if grid[i][j] == 0:
                    continue
                
                res += max(grid[i][j] - grid[i - 1][j], 0) if i - 1 >= 0 else grid[i][j]
                res += max(grid[i][j] - grid[i + 1][j], 0) if i + 1 < N else grid[i][j]
                res += max(grid[i][j] - grid[i][j - 1], 0) if j - 1 >= 0 else grid[i][j]
                res += max(grid[i][j] - grid[i][j + 1], 0) if j + 1 < N else grid[i][j]
                
                res += 2
        
        return res
```



# [912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

难度 中等

给你一个整数数组 `nums`，请你将该数组升序排列。



**示例 1：**

```
输入：nums = [5,2,3,1]
输出：[1,2,3,5]
```

**示例 2：**

```
输入：nums = [5,1,1,2,0,0]
输出：[0,0,1,1,2,5]
```

 

**提示：**

1. `1 <= nums.length <= 50000`
2. `-50000 <= nums[i] <= 50000`



**解法**

+ 方法一：使用库函数。

+ 方法二：快速排序。时间复杂度：平均 $O(N \log N)$ ，最坏 $O(N^2)$ ，空间复杂度：$O(1)$ 。

  

**代码**

```python
# 方法一
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        return nums

# 方法二
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        self.quickSort(nums, 0, len(nums) - 1)
        return nums

    def quickSortPart(self, nums, l, r): 
        pivot = r
        r -= 1
        if l == r:
            if nums[l] > nums[pivot]:
                nums[l], nums[pivot] = nums[pivot], nums[l]

        while l < r:
            while l < r and nums[l] <= nums[pivot]:
                l += 1
            
            if l == r:
                if nums[l] > nums[pivot]:
                    nums[l], nums[pivot] = nums[pivot], nums[l]
                else:
                    l += 1
                    nums[l], nums[pivot] = nums[pivot], nums[l]
                break
                    
            while l < r and nums[r] > nums[pivot]:
                r -= 1
            
            if l == r:
                nums[l], nums[pivot] = nums[pivot], nums[l]
                break

            nums[l], nums[r] = nums[r], nums[l]
        return l


    def quickSort(self, nums, l, r):
        if l >= r:
            return
        mid = self.quickSortPart(nums, l, r)
        self.quickSort(nums, l, mid - 1)
        self.quickSort(nums, mid + 1, r)
```

```cpp
// 方法一
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        return nums;
    }
};
```



# [914. 卡牌分组](https://leetcode-cn.com/problems/x-of-a-kind-in-a-deck-of-cards/)

难度 简单

给定一副牌，每张牌上都写着一个整数。

此时，你需要选定一个数字 `X`，使我们可以将整副牌按下述规则分成 1 组或更多组：

- 每组都有 `X` 张牌。
- 组内所有的牌上都写着相同的整数。

仅当你可选的 `X >= 2` 时返回 `true`。

 

**示例 1：**

```
输入：[1,2,3,4,4,3,2,1]
输出：true
解释：可行的分组是 [1,1]，[2,2]，[3,3]，[4,4]
```

**示例 2：**

```
输入：[1,1,1,2,2,2,3,3]
输出：false
解释：没有满足要求的分组。
```

**示例 3：**

```
输入：[1]
输出：false
解释：没有满足要求的分组。
```

**示例 4：**

```
输入：[1,1]
输出：true
解释：可行的分组是 [1,1]
```

**示例 5：**

```
输入：[1,1,2,2,2,2]
输出：true
解释：可行的分组是 [1,1]，[2,2]，[2,2]
```


**提示：**

1. `1 <= deck.length <= 10000`
2. `0 <= deck[i] < 10000`



**解法**

计算每个数组出现次数的最大公约数，与2作比较。时间复杂度：$O(N \log C)$ 其中 $N$ 是卡牌的个数， $C$ 是数组中数的范围，求两个数最大公约数的复杂度是 $O(\log C)$，空间复杂度：$O(N)$ 。



**代码**

``` python
# 代码一
import math
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        if not deck:
            return False

        d = {}
        for i in deck:
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
        
        count = list(d.values())
        
        res = count[0]
        for i in range(1, len(count)):
            res = math.gcd(res, count[i])
            if res == 1:
                break
        
        return res >= 2


# 代码二
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        vals = collections.Counter(deck).values()

        from fractions import gcd
        return reduce(gcd, vals) >= 2
```



# [945. 使数组唯一的最小增量](https://leetcode-cn.com/problems/minimum-increment-to-make-array-unique/)

难度 中等

给定整数数组 A，每次 *move* 操作将会选择任意 `A[i]`，并将其递增 `1`。

返回使 `A` 中的每个值都是唯一的最少操作次数。

**示例 1:**

```
输入：[1,2,2]
输出：1
解释：经过一次 move 操作，数组将变为 [1, 2, 3]。
```

**示例 2:**

```
输入：[3,2,1,2,1,7]
输出：6
解释：经过 6 次 move 操作，数组将变为 [3, 4, 1, 2, 5, 7]。
可以看出 5 次或 5 次以下的 move 操作是不能让数组的每个值唯一的。
```

**提示：**

1. `0 <= A.length <= 40000`
2. `0 <= A[i] < 40000`



**解法**

+ 方法一：使用数组存储每个数字出现的次数，若一个数字出现多次，则向后查找没有出现过的数字的位置，得到需要加的值。（超时）时间复杂度：$O(n^2)$，空间复杂度：$O(n)$ 。
+ 方法二：对数组进行排序，遇到重复的数字，先将结果减去该数值。若遇到相邻的两个数字之间有空隙可以插入数字，则将之前重复的数字插到这里，结果加上该处的值。时间复杂度：$O(n \log n)$，空间复杂度：$O(n)$  （python 中 `sort` 的空间复杂度）。



**代码**

```python
# 方法一
class Solution:
    def minIncrementForUnique(self, A: List[int]) -> int:
        A.sort()
        A.append(80005)
        ans = 0
        taken = 0

        for i in range(1, len(A)):
            if A[i - 1] == A[i]:
                taken += 1
                ans -= A[i]
            else:
                give = min(taken, A[i] - A[i - 1] - 1)
                ans += give * A[i - 1] + give * (give + 1) // 2
                taken -= give

        return ans


# 方法二
class Solution:
    def minIncrementForUnique(self, A: List[int]) -> int:
        record = [0] * 80002
        for i in A:
            record[i] += 1

        move = 0
        for i in range(0, len(record)):
            while record[i] > 1:
                m = 1
                while record[i + m] != 0:
                    m += 1
                record[i + m] += 1
                move += m 
                record[i] -= 1
        
        return move
```



# [974. 和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k/)

难度 中等

给定一个整数数组 `A`，返回其中元素之和可被 `K` 整除的（连续、非空）子数组的数目。

 

**示例：**

```
输入：A = [4,5,0,-2,-3,1], K = 5
输出：7
解释：
有 7 个子数组满足其元素之和可被 K = 5 整除：
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
```

 

**提示：**

1. `1 <= A.length <= 30000`
2. `-10000 <= A[i] <= 10000`
3. `2 <= K <= 10000`



**解法**

前缀和+哈希表。 前缀和：令 $P[i] = A[0] + A[1] + ... + A[i]$ 。 对于哈希表中的每个键值对 $(x, c_x)$，表示前缀和模 $K$ 后的值 $x$ 出现了 $c_x$ 次。 那么这些出现的位置两两之间都可以构成可被 $K$ 整除的连续子数组。时间复杂度： $O(N)$ ，空间复杂度： $O(\min(N, K))$ 。



**代码**

```python
# 官方题解
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        record = {0: 1}
        total = 0
        for elem in A:
            total += elem
            modulus = total % K
            record[modulus] = record.get(modulus, 0) + 1
        
        ans = 0
        for x, cx in record.items():
            ans += cx * (cx - 1) // 2
        return ans
```



# [983. 最低票价](https://leetcode-cn.com/problems/minimum-cost-for-tickets/)

难度 中等

在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 `days` 的数组给出。每一项是一个从 `1` 到 `365` 的整数。

火车票有三种不同的销售方式：

- 一张为期一天的通行证售价为 `costs[0]` 美元；
- 一张为期七天的通行证售价为 `costs[1]` 美元；
- 一张为期三十天的通行证售价为 `costs[2]` 美元。

通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。

返回你想要完成在给定的列表 `days` 中列出的每一天的旅行所需要的最低消费。

 

**示例 1：**

```
输入：days = [1,4,6,7,8,20], costs = [2,7,15]
输出：11
解释： 
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划：
在第 1 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 1 天生效。
在第 3 天，你花了 costs[1] = $7 买了一张为期 7 天的通行证，它将在第 3, 4, ..., 9 天生效。
在第 20 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 20 天生效。
你总共花了 $11，并完成了你计划的每一天旅行。
```

**示例 2：**

```
输入：days = [1,2,3,4,5,6,7,8,9,10,30,31], costs = [2,7,15]
输出：17
解释：
例如，这里有一种购买通行证的方法，可以让你完成你的旅行计划： 
在第 1 天，你花了 costs[2] = $15 买了一张为期 30 天的通行证，它将在第 1, 2, ..., 30 天生效。
在第 31 天，你花了 costs[0] = $2 买了一张为期 1 天的通行证，它将在第 31 天生效。 
你总共花了 $17，并完成了你计划的每一天旅行。
```

 

**提示：**

1. `1 <= days.length <= 365`
2. `1 <= days[i] <= 365`
3. `days` 按顺序严格递增
4. `costs.length == 3`
5. `1 <= costs[i] <= 1000`



**解法**

动态规划。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:

        @lru_cache()
        def dp(day):
            if day > 365:
                return 0
            elif day in days:
                return min(dp(day + ticketDay) + ticketCost 
                                for ticketDay, ticketCost in zip([1, 7, 30], costs) )
            else:
                return dp(day + 1)

        return dp(1)
```



# [990. 等式方程的可满足性](https://leetcode-cn.com/problems/satisfiability-of-equality-equations/)

难度 中等

给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 `equations[i]` 的长度为 `4`，并采用两种不同的形式之一：`"a==b"` 或 `"a!=b"`。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。

只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 `true`，否则返回 `false`。 

 



**示例 1：**

```
输入：["a==b","b!=a"]
输出：false
解释：如果我们指定，a = 1 且 b = 1，那么可以满足第一个方程，但无法满足第二个方程。没有办法分配变量同时满足这两个方程。
```

**示例 2：**

```
输出：["b==a","a==b"]
输入：true
解释：我们可以指定 a = 1 且 b = 1 以满足满足这两个方程。
```

**示例 3：**

```
输入：["a==b","b==c","a==c"]
输出：true
```

**示例 4：**

```
输入：["a==b","b!=c","c==a"]
输出：false
```

**示例 5：**

```
输入：["c==c","b==d","x!=z"]
输出：true
```

 

**提示：**

1. `1 <= equations.length <= 500`
2. `equations[i].length == 4`
3. `equations[i][0]` 和 `equations[i][3]` 是小写字母
4. `equations[i][1]` 要么是 `'='`，要么是 `'!'`
5. `equations[i][2]` 是 `'='`



**解法**

构建并查集。时间复杂度：$O(N + C *\log C)$ ，空间复杂度： $O(C)$ 。  N 是 `equations` 中的方程数量，C 是变量的总数（26）。 



**代码**

```python
class Solution:

    class UnionFind:
        def __init__(self):
            self.parent = list(range(26))
        
        def find(self, index):
            if index == self.parent[index]:
                return index
            self.parent[index] = self.find(self.parent[index])
            return self.parent[index]
        
        def union(self, index1, index2):
            self.parent[self.find(index1)] = self.find(index2)


    def equationsPossible(self, equations: List[str]) -> bool:
        uf = Solution.UnionFind()
        for st in equations:
            if st[1] == "=":
                index1 = ord(st[0]) - ord("a")
                index2 = ord(st[3]) - ord("a")
                uf.union(index1, index2)
        for st in equations:
            if st[1] == "!":
                index1 = ord(st[0]) - ord("a")
                index2 = ord(st[3]) - ord("a")
                if uf.find(index1) == uf.find(index2):
                    return False
        return True
```



# [994. 腐烂的橘子](https://leetcode-cn.com/problems/rotting-oranges/)

难度 简单

在给定的网格中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，任何与腐烂的橘子（在 4 个正方向上）相邻的新鲜橘子都会腐烂。

返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`。

**提示：**

1. `1 <= grid.length <= 10`
2. `1 <= grid[0].length <= 10`
3. `grid[i][j]` 仅为 `0`、`1` 或 `2`



**解法**

广度优先遍历。时间复杂度：$O(mn)$，空间复杂度：$O(mn)$



**代码**

``` cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        // int** dis = new int*[row];
        // for(int i = 0; i < row; i++)
        //     dis[i] = new int[col];
        int dis[10][10];
        memset(dis, -1, sizeof(dis));
        
        int dir_x[4]={0, 1, 0, -1};
        int dir_y[4]={1, 0, -1, 0};

        queue<pair<int, int>> q;

        int count = 0;
        for(int i = 0; i < row; i++)
            for(int j = 0; j < col; j++)
            {
                if(grid[i][j] == 2)
                {
                    q.push(make_pair(i, j));
                    dis[i][j] = 0;
                }
                if(grid[i][j] == 1)
                    count ++;
            }
        
        int ans = 0;
        while(!q.empty())
        {
            pair<int, int> point = q.front();
            q.pop();

            for(int i = 0; i < 4; i++)
            {
                int x = point.first + dir_x[i];
                int y = point.second + dir_y[i];
                if(x < 0 || x >= row || y < 0 || y >= col || dis[x][y] != -1 || grid[x][y] == 0)
                    continue;
                dis[x][y] = dis[point.first][point.second] + 1;
                q.push(make_pair(x, y));
                if(grid[x][y] == 1)
                {
                    count--;
                    if(!count)
                    {
                        ans = dis[x][y];
                        break;
                    }
                }
            }
        }

        // delete dis;
        return count? -1 : ans;
    }
};
```

```python
class Solution(object):
    def orangesRotting(self, grid):
        R, C = len(grid), len(grid[0])

        # queue - all starting cells with rotting oranges
        queue = collections.deque()
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val == 2:
                    queue.append((r, c, 0))

        def neighbors(r, c):
            for nr, nc in ((r-1,c),(r,c-1),(r+1,c),(r,c+1)):
                if 0 <= nr < R and 0 <= nc < C:
                    yield nr, nc

        d = 0
        while queue:
            r, c, d = queue.popleft()
            for nr, nc in neighbors(r, c):
                if grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    queue.append((nr, nc, d+1))

        if any(1 in row for row in grid):
            return -1
        return d

# https://leetcode-cn.com/problems/rotting-oranges/solution/fu-lan-de-ju-zi-by-leetcode-solution/
```



# [999. 车的可用捕获量](https://leetcode-cn.com/problems/available-captures-for-rook/)

难度 简单

在一个 8 x 8 的棋盘上，有一个白色车（rook）。也可能有空方块，白色的象（bishop）和黑色的卒（pawn）。它们分别以字符 “R”，“.”，“B” 和 “p” 给出。大写字符表示白棋，小写字符表示黑棋。

车按国际象棋中的规则移动：它选择四个基本方向中的一个（北，东，西和南），然后朝那个方向移动，直到它选择停止、到达棋盘的边缘或移动到同一方格来捕获该方格上颜色相反的卒。另外，车不能与其他友方（白色）象进入同一个方格。

返回车能够在一次移动中捕获到的卒的数量。


**示例 1：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1253_example_1_improved.PNG)

```
输入：[[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","R",".",".",".","p"],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
输出：3
解释：
在本例中，车能够捕获所有的卒。
```

**示例 2：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1253_example_2_improved.PNG)

```
输入：[[".",".",".",".",".",".",".","."],[".","p","p","p","p","p",".","."],[".","p","p","B","p","p",".","."],[".","p","B","R","B","p",".","."],[".","p","p","B","p","p",".","."],[".","p","p","p","p","p",".","."],[".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".","."]]
输出：0
解释：
象阻止了车捕获任何卒。
```

**示例 3：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/23/1253_example_3_improved.PNG)

```
输入：[[".",".",".",".",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".","p",".",".",".","."],["p","p",".","R",".","p","B","."],[".",".",".",".",".",".",".","."],[".",".",".","B",".",".",".","."],[".",".",".","p",".",".",".","."],[".",".",".",".",".",".",".","."]]
输出：3
解释： 
车可以捕获位置 b5，d6 和 f5 的卒。
```

 

**提示：**

1. `board.length == board[i].length == 8`
2. `board[i][j]` 可以是 `'R'`，`'.'`，`'B'` 或 `'p'`
3. 只有一个格子上存在 `board[i][j] == 'R'`



**解法**

遍历数组，找到车的位置，然后模拟车上下左右进行移动。时间复杂度：$O(n^2)$，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def numRookCaptures(self, board: List[List[str]]) -> int:
        loc = None      # 车的位置
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    loc = (i, j)
                    break
            if loc:
                break

        if not loc:
            return 0

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        res = 0
        for i in range(0, 4):
            move = 1
            while True:
                x = loc[0] + dx[i] * move
                y = loc[1] + dy[i] * move

                if x < 0 or x > 7 or y < 0 or y > 7:
                    break

                if board[x][y] == 'p':
                    res += 1
                    break

                if board[x][y] == 'B':
                    break

                move += 1
        
        return res
```



# [1013. 将数组分成和相等的三个部分](https://leetcode-cn.com/problems/partition-array-into-three-parts-with-equal-sum/)

难度 简单

给你一个整数数组 `A`，只有可以将其划分为三个和相等的非空部分时才返回 `true`，否则返回 `false`。

形式上，如果可以找出索引 `i+1 < j` 且满足 `(A[0] + A[1] + ... + A[i] == A[i+1] + A[i+2] + ... + A[j-1] == A[j] + A[j-1] + ... + A[A.length - 1])` 就可以将数组三等分。

 

**示例 1：**

```
输出：[0,2,1,-6,6,-7,9,1,2,0,1]
输出：true
解释：0 + 2 + 1 = -6 + 6 - 7 + 9 + 1 = 2 + 0 + 1
```

**示例 2：**

```
输入：[0,2,1,-6,6,7,9,-1,2,0,1]
输出：false
```

**示例 3：**

```
输入：[3,3,6,5,-2,2,5,1,-9,4]
输出：true
解释：3 + 3 = 6 = 5 - 2 + 2 + 5 + 1 - 9 + 4
```

 

**提示：**

1. `3 <= A.length <= 50000`
2. `-10^4 <= A[i] <= 10^4`



**解法**

分成三部分后，每部分的和为数组和的三分之一。使用指针从头到尾进行累加，确定分割点位置。

时间复杂度：$O(n)$，空间复杂度：$O(1)$



**代码**

```python
class Solution:
    def canThreePartsEqualSum(self, A: List[int]) -> bool:
        s = sum(A)

        if s % 3 != 0:
            return False
        
        index = 1
        t_sum = A[0]       # 当前部分的和
        while index < len(A) and t_sum != s // 3:
            t_sum += A[index]
            index += 1
        
        if t_sum == s // 3:
            t_sum = A[index]
            index += 1
            while index < len(A) and t_sum != s // 3:
                t_sum += A[index]
                index += 1
            
            if index < len(A) and sum(A[index:]) == s // 3:
                return True 
        
        return False
```



# [1014. 最佳观光组合](https://leetcode-cn.com/problems/best-sightseeing-pair/)

给定正整数数组 `A`，`A[i]` 表示第 `i` 个观光景点的评分，并且两个景点 `i` 和 `j` 之间的距离为 `j - i`。

一对景点（`i < j`）组成的观光组合的得分为（`A[i] + A[j] + i - j`）：景点的评分之和**减去**它们两者之间的距离。

返回一对观光景点能取得的最高分。

 

**示例：**

```
输入：[8,1,5,2,6]
输出：11
解释：i = 0, j = 2, A[i] + A[j] + i - j = 8 + 5 + 0 - 2 = 11
```

 

**提示：**

1. `2 <= A.length <= 50000`
2. `1 <= A[i] <= 1000`



**解法**

将表达式视为 `A[i] + i` 和 `A[j] -j` 两部分，这两部分的值都只与元素自己有关，遍历找出最大值即可。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        maxi = A[0] + 0
        ans = 0

        for i in range(1, len(A)):
            ans = max(ans, maxi + A[i] - i)
            maxi = max(maxi, A[i] + i )
        
        return ans
```



# [1028. 从先序遍历还原二叉树](https://leetcode-cn.com/problems/recover-a-tree-from-preorder-traversal/)

难度 困难

我们从二叉树的根节点 `root` 开始进行深度优先搜索。

在遍历中的每个节点处，我们输出 `D` 条短划线（其中 `D` 是该节点的深度），然后输出该节点的值。（*如果节点的深度为 `D`，则其直接子节点的深度为 `D + 1`。根节点的深度为 `0`）。*

如果节点只有一个子节点，那么保证该子节点为左子节点。

给出遍历输出 `S`，还原树并返回其根节点 `root`。

 

**示例 1：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/12/recover-a-tree-from-preorder-traversal.png)**

```
输入："1-2--3--4-5--6--7"
输出：[1,2,5,3,4,6,7]
```

**示例 2：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/12/screen-shot-2019-04-10-at-114101-pm.png)**

```
输入："1-2--3---4-5--6---7"
输出：[1,2,5,3,null,6,null,4,null,7]
```

**示例 3：**

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/04/12/screen-shot-2019-04-10-at-114955-pm.png)

```
输入："1-401--349---90--88"
输出：[1,401,null,349,88,90]
```

 

**提示：**

- 原始树中的节点数介于 `1` 和 `1000` 之间。
- 每个节点的值介于 `1` 和 `10 ^ 9` 之间。



**解法**

使用列表保存已完成的树的节点，对字符串进行遍历构建树。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 



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
    def recoverFromPreorder(self, S: str) -> TreeNode:
        path, pos = list(), 0
        while pos < len(S):
            level = 0
            while S[pos] == '-':
                level += 1
                pos += 1
            value = 0
            while pos < len(S) and S[pos].isdigit():
                value = value * 10 + (ord(S[pos]) - ord('0'))
                pos += 1
            node = TreeNode(value)
            if level == len(path):
                if path:
                    path[-1].left = node
            else:
                path = path[:level]
                path[-1].right = node
            path.append(node)
        return path[0]

```



# [1071. 字符串的最大公因子](https://leetcode-cn.com/problems/greatest-common-divisor-of-strings/)

难度 简单 

对于字符串 `S` 和 `T`，只有在 `S = T + ... + T`（`T` 与自身连接 1 次或多次）时，我们才认定 “`T` 能除尽 `S`”。

返回最长字符串 `X`，要求满足 `X` 能除尽 `str1` 且 `X` 能除尽 `str2`。

 

**示例 1：**

```
输入：str1 = "ABCABC", str2 = "ABC"
输出："ABC"
```

**示例 2：**

```
输入：str1 = "ABABAB", str2 = "ABAB"
输出："AB"
```

**示例 3：**

```
输入：str1 = "LEET", str2 = "CODE"
输出：""
```

 

**提示：**

1. `1 <= str1.length <= 1000`
2. `1 <= str2.length <= 1000`
3. `str1[i]` 和 `str2[i]` 为大写英文字母



**解法**

辗转相减。时间复杂度：$O(\max(m, n))$ $m, n$为字符串长度，空间复杂度：$O(1)$



**代码**

``` python
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:

        if len(str1) > len(str2):
            str_l = str1
            str_s = str2
        else:
            str_l = str2
            str_s = str1

        match = True

        while len(str_s) > 0:
            if str_l == str_s:
                break

            if str_l[:len(str_s)] != str_s:
                match = False
                break

            str_l = str_l[len(str_s):]

            if len(str_l) < len(str_s):
                str_l, str_s = str_s, str_l

        if match:
            return str_s
        else:
            return ""
```



#  [1095. 山脉数组中查找目标值](https://leetcode-cn.com/problems/find-in-mountain-array/)

难度 困难

（这是一个 **交互式问题** ）

给你一个 **山脉数组** `mountainArr`，请你返回能够使得 `mountainArr.get(index)` **等于** `target` **最小** 的下标 `index` 值。

如果不存在这样的下标 `index`，就请返回 `-1`。

 

何为山脉数组？如果数组 `A` 是一个山脉数组的话，那它满足如下条件：

**首先**，`A.length >= 3`

**其次**，在 `0 < i < A.length - 1` 条件下，存在 `i` 使得：

- `A[0] < A[1] < ... A[i-1] < A[i]`
- `A[i] > A[i+1] > ... > A[A.length - 1]`

 

你将 **不能直接访问该山脉数组**，必须通过 `MountainArray` 接口来获取数据：

- `MountainArray.get(k)` - 会返回数组中索引为`k` 的元素（下标从 0 开始）
- `MountainArray.length()` - 会返回该数组的长度

 

**注意：**

对 `MountainArray.get` 发起超过 `100` 次调用的提交将被视为错误答案。此外，任何试图规避判题系统的解决方案都将会导致比赛资格被取消。

为了帮助大家更好地理解交互式问题，我们准备了一个样例 “**答案**”：https://leetcode-cn.com/playground/RKhe3ave，请注意这 **不是一个正确答案**。



 

**示例 1：**

```
输入：array = [1,2,3,4,5,3,1], target = 3
输出：2
解释：3 在数组中出现了两次，下标分别为 2 和 5，我们返回最小的下标 2。
```

**示例 2：**

```
输入：array = [0,1,2,4,2,1], target = 3
输出：-1
解释：3 在数组中没有出现，返回 -1。
```

 

**提示：**

- `3 <= mountain_arr.length() <= 10000`
- `0 <= target <= 10^9`
- `0 <= mountain_arr.get(index) <= 10^9`



**解法**

二分法找到山顶，再在山两边进行二分查找。时间复杂度： $O(\log N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        n =  mountain_arr.length()

        left = 0
        right = n - 1
        while left < right:
            mid = (left + right) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
                left = mid + 1
            else:
                right = mid
        mid += 1

        left = 0
        right = mid
        while left <= right:
            m = (left + right) // 2
            val = mountain_arr.get(m)
            if val == target:
                return m
            elif val < target:
                left = m + 1
            else:
                right = m - 1
        
        left = mid
        right = n - 1
        while left <= right:
            m = (left + right) // 2
            val = mountain_arr.get(m)
            if val == target:
                return m
            elif val > target:
                left = m + 1
            else:
                right = m - 1
        
        return -1
```



# [1103. 分糖果 II](https://leetcode-cn.com/problems/distribute-candies-to-people/)

难度 简单

排排坐，分糖果。

我们买了一些糖果 candies，打算把它们分给排好队的 n = num_people 个小朋友。

给第一个小朋友 1 颗糖果，第二个小朋友 2 颗，依此类推，直到给最后一个小朋友 n 颗糖果。

然后，我们再回到队伍的起点，给第一个小朋友 n + 1 颗糖果，第二个小朋友 n + 2 颗，依此类推，直到给最后一个小朋友 2 * n 颗糖果。

重复上述过程（每次都比上一次多给出一颗糖果，当到达队伍终点后再次从队伍起点开始），直到我们分完所有的糖果。注意，就算我们手中的剩下糖果数不够（不比前一次发出的糖果多），这些糖果也会全部发给当前的小朋友。

返回一个长度为 num_people、元素之和为 candies 的数组，以表示糖果的最终分发情况（即 ans[i] 表示第 i 个小朋友分到的糖果数）。


示例 1：
```
输入：candies = 7, num_people = 4
输出：[1,2,3,1]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0,0]。
第三次，ans[2] += 3，数组变为 [1,2,3,0]。
第四次，ans[3] += 1（因为此时只剩下 1 颗糖果），最终数组变为 [1,2,3,1]。
```
示例 2：
```
输入：candies = 10, num_people = 3
输出：[5,2,3]
解释：
第一次，ans[0] += 1，数组变为 [1,0,0]。
第二次，ans[1] += 2，数组变为 [1,2,0]。
第三次，ans[2] += 3，数组变为 [1,2,3]。
第四次，ans[0] += 4，最终数组变为 [5,2,3]。
```

提示：

+ `1 <= candies <= 10^9`
+ `1 <= num_people <= 1000`



**解法**

+ 方法一：暴力法。模拟发糖，发完为止。时间复杂度：$\mathcal{O}(max(\sqrt{G}, N))$，空间复杂度：$\mathcal{O}(1)$。 $G$为糖果数量，$N$ 为人数。 
+ 方法二：利用等差数列计算可完整发完的轮数，再模拟最后一轮。时间复杂度：$\mathcal{O}(N)$，空间复杂度：$\mathcal{O}(1)$。 
+ 方法三：利用等差数列计算可完整发完的人数，再计算未能完整发完的人数。（官方题解）时间复杂度：$\mathcal{O}(N)$，空间复杂度：$\mathcal{O}(1)$。 



**代码**

``` python
# 方法二
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        loop = 0
        n =  num_people
        while n * (n + 1) / 2 <= candies:
            loop += 1
            n = num_people * (loop + 1)
        
        if loop == 0:
            result = [0 for i in range(num_people)]
            base = 1
        else:
            result = []
            base = loop * (num_people * (loop - 1) + 2) // 2 

            for i in range(0, num_people):
                result.append(base)
                candies -= base
                base += loop
            
            base = num_people * loop + 1
        
        if candies:
            index = 0
            while base < candies:
                result[index] += base
                candies -= base
                base += 1
                index += 1
            result[index] += candies
            
        return result
```



# [1111. 有效括号的嵌套深度](https://leetcode-cn.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/)

难度 中等

**有效括号字符串** 定义：对于每个左括号，都能找到与之对应的右括号，反之亦然。详情参见题末「**有效括号字符串**」部分。

**嵌套深度** `depth` 定义：即有效括号字符串嵌套的层数。详情参见题末「**嵌套深度**」部分。

 

给你一个「有效括号字符串」 `seq`，请你将其分成两个不相交的子序列 `A` 和 `B`，且 `A` 和 `B` 都满足有效括号字符串的定义（注意：`A.length + B.length = seq.length`）。

由于可能存在多种划分方案，请你从中选出 **任意** 一组有效括号字符串 `A` 和 `B`，使 `max(depth(A), depth(B))` 的可能取值最小。其中 `depth(A)` 表示 `A` 的嵌套深度，`depth(B)` 表示 `B` 的嵌套深度。

请你返回一个长度为 `seq.length` 的答案数组 `answer`，编码规则如下：如果 `seq[i]` 是 `A` 的一部分，那么 `answer[i] = 0`。否则，`answer[i] = 1`。即便有多个满足要求的答案存在，你也只需返回 **一个**。

 

**示例 1：**

```
输入：seq = "(()())"
输出：[0,1,1,1,1,0]
```

**示例 2：**

```
输入：seq = "()(())()"
输出：[0,0,0,1,1,0,1,1]
```

 

**提示：**

- `1 <= text.size <= 10000`

 

**有效括号字符串：**

仅由 `"("` 和 `")"` 构成的字符串，对于每个左括号，都能找到与之对应的右括号，反之亦然。

下述几种情况同样属于有效括号字符串：

- 空字符串
- 连接，可以记作 `AB`（`A` 与 `B` 连接），其中 `A` 和 `B` 都是有效括号字符串
- 嵌套，可以记作 `(A)`，其中 `A` 是有效括号字符串

**嵌套深度：**

类似地，我们可以定义任意有效括号字符串 `s` 的 **嵌套深度** `depth(S)`：

- `s` 为空时，`depth("") = 0`
- `s` 为 `A` 与 `B` 连接时，`depth(A + B) = max(depth(A), depth(B))`，其中 `A` 和 `B` 都是有效括号字符串
- `s` 为嵌套情况，`depth("(" + A + ")") = 1 + depth(A)`，其中 A 是有效括号字符串

例如：`""`，`"()()"`，和 `"()(()())"` 都是有效括号字符串，嵌套深度分别为 0，1，2，而 `")("` 和 `"(()"` 都不是有效括号字符串。



**解法**

用栈进行括号匹配。计算当前嵌套深度。 把奇数层的 `(` 分配给 `A`，偶数层的 `(` 分配给 `B` 。时间复杂度：$\mathcal{O}(N)$，空间复杂度：$\mathcal{O}(1)$。 



**代码**

```python
class Solution:
    def maxDepthAfterSplit(self, seq: str) -> List[int]:
        ans = []
        depth = 0
        for ch in seq:
            if ch == '(':
                depth += 1
                ans.append(depth % 2)
            elif ch == ')':
                ans.append(depth % 2)
                depth -= 1
        return ans
```



# [1160. 拼写单词](https://leetcode-cn.com/problems/find-words-that-can-be-formed-by-characters/)

难度 简单

给你一份『词汇表』（字符串数组） `words` 和一张『字母表』（字符串） `chars`。

假如你可以用 `chars` 中的『字母』（字符）拼写出 `words` 中的某个『单词』（字符串），那么我们就认为你掌握了这个单词。

注意：每次拼写时，`chars` 中的每个字母都只能用一次。

返回词汇表 `words` 中你掌握的所有单词的 **长度之和**。

 

**示例 1：**

```
输入：words = ["cat","bt","hat","tree"], chars = "atach"
输出：6
解释： 
可以形成字符串 "cat" 和 "hat"，所以答案是 3 + 3 = 6。
```

**示例 2：**

```
输入：words = ["hello","world","leetcode"], chars = "welldonehoneyr"
输出：10
解释：
可以形成字符串 "hello" 和 "world"，所以答案是 5 + 5 = 10。
```

 

**提示：**

1. `1 <= words.length <= 1000`
2. `1 <= words[i].length, chars.length <= 100`
3. 所有字符串中都仅包含小写英文字母



**解法**

+ 方法一：将字母保存至列表。遍历每个单词，遍历该单词的每个字母，若字母在拷贝的字母表中，则移除该字母一次；若不在，则无法拼写。时间复杂度：$O(nm)$ ，空间复杂度：$O(m)$ ， $n$ 为单词表中所有单词总长度， $m$ 为字母表长度。
+ 方法二：将字母及其个数保存至字典。遍历每个单词，将其字母及其个数保存至字典。比较两个字典中单词及其个数，判断是否可以拼写。时间复杂度：$O(n+m)$ ，空间复杂度：$O(m)$ ， $n$ 为单词表中所有单词总长度， $m$ 为字母表长度。



**代码**

```python
# 方法一
import copy
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        char = []
        for c in chars:
            char.append(c)
        
        result = 0
        for word in words:
            if len(word) > len(char):
                continue
            
            ch = copy.copy(char)
            spell = True
            for w in word:
                if w in ch:
                    ch.remove(w)
                else:
                    spell = False
                    break
            if spell:
                result += len(word)
        
        return result

# 方法二
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        chars_cnt = collections.Counter(chars)
        ans = 0
        for word in words:
            word_cnt = collections.Counter(word)
            for c in word_cnt:
                if chars_cnt[c] < word_cnt[c]:
                    break
            else:
                ans += len(word)
        return ans
```



# [1162. 地图分析](https://leetcode-cn.com/problems/as-far-from-land-as-possible/)

难度 中等

你现在手里有一份大小为 N x N 的『地图』（网格） `grid`，上面的每个『区域』（单元格）都用 `0` 和 `1` 标记好了。其中 `0` 代表海洋，`1` 代表陆地，你知道距离陆地区域最远的海洋区域是是哪一个吗？请返回该海洋区域到离它最近的陆地区域的距离。

我们这里说的距离是『曼哈顿距离』（ Manhattan Distance）：`(x0, y0)` 和 `(x1, y1)` 这两个区域之间的距离是 `|x0 - x1| + |y0 - y1|` 。

如果我们的地图上只有陆地或者海洋，请返回 `-1`。

 

**示例 1：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/08/17/1336_ex1.jpeg)**

```
输入：[[1,0,1],[0,0,0],[1,0,1]]
输出：2
解释： 
海洋区域 (1, 1) 和所有陆地区域之间的距离都达到最大，最大距离为 2。
```

**示例 2：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/08/17/1336_ex2.jpeg)**

```
输入：[[1,0,0],[0,0,0],[0,0,0]]
输出：4
解释： 
海洋区域 (2, 2) 和所有陆地区域之间的距离都达到最大，最大距离为 4。
```

 

**提示：**

1. `1 <= grid.length == grid[0].length <= 100`
2. `grid[i][j]` 不是 `0` 就是 `1`



**解法**

+ 方法一：广度优先遍历。（超时）时间复杂度：$O(N^4)$，空间复杂度：$O(N^2)$ 。
+ 方法二：多源广度优先遍历。将陆地作为源点集，进行广度优先遍历。时间复杂度：$O(N^2)$，空间复杂度：$O(N^2)$ 。
+ 方法三：动态规划。  对于每个海洋区域，离它最近的陆地区域到它的路径要么从上方或者左方来，要么从右方或者下方来。 做两次动态规划，第一次从左上到右下，第二次从右下到左上。时间复杂度：$O(N^2)$，空间复杂度：$O(N^2)$ 。



**代码**

```python
# 方法一
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        self.size = len(grid)
        self.grid = grid

        res = -1
        dis = 0
        for i in range(0, self.size):
            for j in range(0, self.size):
                if grid[i][j] == 0:
                    dis = self.BFS(i, j)
                    res = max(res, dis)
                    if dis == -1:
                        break
            if dis == -1:
                break
        
        return res
        

    def BFS(self, x, y):
        li = [[x, y, 0]]
        visited = [[False] * self.size for i in range(self.size)]
        
        dis_x = [-1, 1, 0, 0]
        dis_y = [0, 0, -1, 1]
        index = 0
        land = False

        while index < len(li):
            x = li[index][0]
            y = li[index][1]
            d = li[index][2]

            index += 1
            
            for i in range(0, 4):
                if x + dis_x[i] >= 0 and x + dis_x[i] < self.size:
                    if y + dis_y[i] >= 0 and y + dis_y[i] < self.size:
                        if self.grid[x + dis_x[i]][y + dis_y[i]] == 0:
                            if not visited[x + dis_x[i]][y + dis_y[i]]:
                                li.append([x + dis_x[i], y + dis_y[i], d + 1])
                                visited[x + dis_x[i]][y + dis_y[i]] = True
                        else:
                            land = True
                            break

            if land:
                break

        if land:
            return d + 1
        else:
            return -1

                
# 方法二
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)

        INF = 10000
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        d = [[INF] * n for i in range(n)]

        import queue
        q = queue.Queue()

        # 将陆地作为源点
        for i in range(0, n):
            for j in range(0, n):
                if grid[i][j] == 1:
                    d[i][j] = 0
                    q.put([i, j])
        
        while not q.empty():
            x, y = q.get()
            for i in range(0, 4):
                xx = x + dx[i]
                yy = y + dy[i]
                if xx >= 0 and xx < n and yy >= 0 and yy < n:
                    if d[xx][yy] > d[x][y] + 1:
                        d[xx][yy] =  d[x][y] + 1
                        q.put([xx, yy])
        
        ans = -1
        for i in range(0, n):
            for j in range(0, n):
                if grid[i][j] == 0:
                    ans = max(ans, d[i][j])

        return -1 if ans == INF else ans
    
    
# 方法三
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)

        INF = 10000
        d = [[INF] * n for i in range(n)]

        for i in range(0, n):
            for j in range(0, n):
                d[i][j] = 0 if grid[i][j] == 1 else INF

        for i in range(0, n):
            for j in range(0, n):
                if grid[i][j] == 0:
                    if i - 1 >= 0:
                        d[i][j] = min(d[i - 1][j] + 1, d[i][j])
                    if j - 1 >= 0:
                        d[i][j] = min(d[i][j - 1] + 1, d[i][j])
        
        for i in range(n-1 , -1, -1):
            for j in range(n-1 , -1, -1):
                if grid[i][j] == 0:
                    if i + 1 < n:
                        d[i][j] = min(d[i + 1][j] + 1, d[i][j])
                    if j + 1 < n:
                        d[i][j] = min(d[i][j + 1] + 1, d[i][j])
        
        ans = -1

        for i in range(0, n):
            for j in range(0, n):
                if grid[i][j] == 0:
                    ans = max(ans, d[i][j])

        return ans if ans != INF else -1
```



# [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/)

难度 中等

给你一个整数数组 `nums` 和一个整数 `k`。

如果某个 **连续** 子数组中恰好有 `k` 个奇数数字，我们就认为这个子数组是「**优美子数组**」。

请返回这个数组中「优美子数组」的数目。

 

**示例 1：**

```
输入：nums = [1,1,2,1,1], k = 3
输出：2
解释：包含 3 个奇数的子数组是 [1,1,2,1] 和 [1,2,1,1] 。
```

**示例 2：**

```
输入：nums = [2,4,6], k = 1
输出：0
解释：数列中不包含任何奇数，所以不存在优美子数组。
```

**示例 3：**

```
输入：nums = [2,2,2,1,2,2,1,2,2,2], k = 2
输出：16
```

 

**提示：**

- `1 <= nums.length <= 50000`
- `1 <= nums[i] <= 10^5`
- `1 <= k <= nums.length`



**解法**

记录每个奇数的下标，则两个奇数之间的数均为偶数。可以得出满足 $l\in (\textit{odd}[i-1],\textit{odd}[i])$ 且 $r\in [\textit{odd}[i+k-1],\textit{odd}[i+k])$ 条件的子数组 $[l,r]$ 里的奇数个数为 $k$ 个。时间复杂度：$\mathcal{O}(N)$，空间复杂度：$\mathcal{O}(N)$。 



**代码**

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)

        odd = [-1]
        for i in range(len(nums)):
            if nums[i] % 2 == 1:
                odd.append(i)
        odd.append(n)

        ans = 0
        for i in range(1, len(odd) - k):
            ans += (odd[i] - odd[i - 1]) * (odd[i + k] - odd[i + k - 1])

        return ans 
```



# [1300. 转变数组后最接近目标值的数组和](https://leetcode-cn.com/problems/sum-of-mutated-array-closest-to-target/)

难度 中等

给你一个整数数组 `arr` 和一个目标值 `target` ，请你返回一个整数 `value` ，使得将数组中所有大于 `value` 的值变成 `value` 后，数组的和最接近 `target` （最接近表示两者之差的绝对值最小）。

如果有多种使得和最接近 `target` 的方案，请你返回这些整数中的最小值。

请注意，答案不一定是 `arr` 中的数字。

 

**示例 1：**

```
输入：arr = [4,9,3], target = 10
输出：3
解释：当选择 value 为 3 时，数组会变成 [3, 3, 3]，和为 9 ，这是最接近 target 的方案。
```

**示例 2：**

```
输入：arr = [2,3,5], target = 10
输出：5
```

**示例 3：**

```
输入：arr = [60864,25176,27249,21296,20204], target = 56803
输出：11361
```

 

**提示：**

- `1 <= arr.length <= 10^4`
- `1 <= arr[i], target <= 10^5`



**解法**

对数组进行排序。在0和数组最大值之间进行二分查找，找到一个 `value` 使得结果小于且最接近 `target` ，则 `value + 1` 的结果大于 `target` ，比较 `value` 和 `value + 1` 得到的结果，可得到答案。时间复杂度： $O(N \log N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def findBestValue(self, arr: List[int], target: int) -> int:
        arr.sort()
        n = len(arr)
        prefix = [0]
        for num in arr:
            prefix.append(prefix[-1] + num)
        
        l, r, ans = 0, max(arr), -1
        while l <= r:
            mid = (l + r) // 2
            it = bisect.bisect_left(arr, mid)
            cur = prefix[it] + (n - it) * mid
            if cur <= target:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1

        def check(x):
            return sum(x if num >= x else num for num in arr)
        
        choose_small = check(ans)
        choose_big = check(ans + 1)
        return ans if abs(choose_small - target) <= abs(choose_big - target) else ans + 1
```



# [1371. 每个元音包含偶数次的最长子字符串](https://leetcode-cn.com/problems/find-the-longest-substring-containing-vowels-in-even-counts/)

难度 中等

给你一个字符串 `s` ，请你返回满足以下条件的最长子字符串的长度：每个元音字母，即 'a'，'e'，'i'，'o'，'u' ，在子字符串中都恰好出现了偶数次。

 

**示例 1：**

```
输入：s = "eleetminicoworoep"
输出：13
解释：最长子字符串是 "leetminicowor" ，它包含 e，i，o 各 2 个，以及 0 个 a，u 。
```

**示例 2：**

```
输入：s = "leetcodeisgreat"
输出：5
解释：最长子字符串是 "leetc" ，其中包含 2 个 e 。
```

**示例 3：**

```
输入：s = "bcbcbc"
输出：6
解释：这个示例中，字符串 "bcbcbc" 本身就是最长的，因为所有的元音 a，e，i，o，u 都出现了 0 次。
```

 

**提示：**

- `1 <= s.length <= 5 x 10^5`
- `s` 只包含小写英文字母。



**解法**

前缀和+状态压缩。 0代表出现了偶数次，1代表出现了奇数次 。使用0/1序列表示在当前位置每个元音字母出现的次数。时间复杂度： $O(N)$ ，空间复杂度： $O(len(S))$ 。



**代码**

```cpp
//官方题解
class Solution {
public:
    int findTheLongestSubstring(string s) {
        int ans = 0, status = 0, n = s.length();
        vector<int> pos(1 << 5, -1);
        pos[0] = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] == 'a') {
                status ^= 1<<0;
            } else if (s[i] == 'e') {
                status ^= 1<<1;
            } else if (s[i] == 'i') {
                status ^= 1<<2;
            } else if (s[i] == 'o') {
                status ^= 1<<3;
            } else if (s[i] == 'u') {
                status ^= 1<<4;
            }
            if (~pos[status]) {
                ans = max(ans, i + 1 - pos[status]);
            } else {
                pos[status] = i + 1;
            }
        }
        return ans;
    }
};
```



# [1431. 拥有最多糖果的孩子](https://leetcode-cn.com/problems/kids-with-the-greatest-number-of-candies/)

难度 简单

给你一个数组 `candies` 和一个整数 `extraCandies` ，其中 `candies[i]` 代表第 `i` 个孩子拥有的糖果数目。

对每一个孩子，检查是否存在一种方案，将额外的 `extraCandies` 个糖果分配给孩子们之后，此孩子有 **最多** 的糖果。注意，允许有多个孩子同时拥有 **最多** 的糖果数目。

 

**示例 1：**

```
输入：candies = [2,3,5,1,3], extraCandies = 3
输出：[true,true,true,false,true] 
解释：
孩子 1 有 2 个糖果，如果他得到所有额外的糖果（3个），那么他总共有 5 个糖果，他将成为拥有最多糖果的孩子。
孩子 2 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
孩子 3 有 5 个糖果，他已经是拥有最多糖果的孩子。
孩子 4 有 1 个糖果，即使他得到所有额外的糖果，他也只有 4 个糖果，无法成为拥有糖果最多的孩子。
孩子 5 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
```

**示例 2：**

```
输入：candies = [4,2,1,1,2], extraCandies = 1
输出：[true,false,false,false,false] 
解释：只有 1 个额外糖果，所以不管额外糖果给谁，只有孩子 1 可以成为拥有糖果最多的孩子。
```

**示例 3：**

```
输入：candies = [12,1,12], extraCandies = 10
输出：[true,false,true]
```

 

**提示：**

- `2 <= candies.length <= 100`
- `1 <= candies[i] <= 100`
- `1 <= extraCandies <= 50`



**解法**

略。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        maxCandy = max(candies)
        ans = []
        for child in candies:
            if maxCandy - child <= extraCandies:
                ans.append(True)
            else:
                ans.append(False)
        return ans
```


