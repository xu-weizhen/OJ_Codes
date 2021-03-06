[toc]



# [257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

难度简单

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明:** 叶子节点是指没有子节点的节点。

**示例:**

```
输入:

   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```



**解法**

深度优先搜索。时间复杂度： $O(n^2)$ ，空间复杂度： $O(n^2)$ ：每个节点需要拷贝 `path` 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:

        ans = []
        path = ''

        def dfs(node, path):
            if node.left is None and node.right is None:
                ans.append(path + str(node.val))
                return
            
            tPath = path + str(node.val) + '->'

            if node.left is not None:
                dfs(node.left, tPath)
            
            if node.right is not None:
                dfs(node.right, tPath)
        
        if not root:
            return ans 
            
        dfs(root, '')
        return ans
```



# [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

难度 中等

给定一个包含 *n* + 1 个整数的数组 *nums*，其数字都在 1 到 *n* 之间（包括 1 和 *n*），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

**示例 1:**

```
输入: [1,3,4,2,2]
输出: 2
```

**示例 2:**

```
输入: [3,1,3,4,2]
输出: 3
```

**说明：**

1. **不能**更改原数组（假设数组是只读的）。
2. 只能使用额外的 *O*(1) 的空间。
3. 时间复杂度小于 *O*(*n*2) 。
4. 数组中只有一个重复的数字，但它可能不止重复出现一次。



**解法**

将数值视为有向图，数值为指向，由于存在相同数字，则该图存在环。使用快慢指针进行查找。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        fast = nums[nums[0]]
        slow = nums[0]

        while fast != slow:
            fast = nums[nums[fast]]
            slow = nums[slow]
        
        slow = 0
        while fast != slow:
            fast = nums[fast]
            slow = nums[slow]
        
        return slow
```



# [289. 生命游戏](https://leetcode-cn.com/problems/game-of-life/)

难度 中等

根据 [百度百科](https://baike.baidu.com/item/生命游戏/2926434?fr=aladdin) ，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。

给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：

1. 如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
2. 如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
3. 如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
4. 如果死细胞周围正好有三个活细胞，则该位置死细胞复活；

根据当前状态，写一个函数来计算面板上所有细胞的下一个（一次更新后的）状态。下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。

 

**示例：**

```
输入： 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
输出：
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]
```

 

**进阶：**

- 你可以使用原地算法解决本题吗？请注意，面板上所有格子需要同时被更新：你不能先更新某些格子，然后使用它们的更新后的值再更新其他格子。
- 本题中，我们使用二维数组来表示面板。原则上，面板是无限的，但当活细胞侵占了面板边界时会造成问题。你将如何解决这些问题？



**解法**

使用新的数字来表示细胞变化的状态，例如 `live -> dead` 标记为 `2` ， `dead -> live` 标记为 `-1` 。遍历每个细胞，确定细胞状态。再遍历一遍将状态改为题目要求。时间复杂度：$O(MN)$ 。空间复杂度：$O(1)$ 。



**代码**

``` python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        row = len(board)
        col = len(board[0])

        # live 1
        # dead 0
        # live -> dead 2
        # dead -> live -1
        for i in range(row):
            for j in range(col):
                count = 0

                if i - 1 >= 0:
                    if j - 1 >= 0 and board[i - 1][j - 1] > 0:
                        count += 1
                    if board[i - 1][j] > 0:
                        count += 1
                    if j + 1 < col and board[i - 1][j + 1] > 0:
                        count += 1
                
                if j - 1 >= 0 and board[i][j - 1] > 0:
                        count += 1
                if j + 1 < col and board[i][j + 1] > 0:
                    count += 1

                if i + 1 < row:
                    if j - 1 >= 0 and board[i + 1][j - 1] > 0:
                        count += 1
                    if board[i + 1][j] > 0:
                        count += 1
                    if j + 1 < col and board[i + 1][j + 1] > 0:
                        count += 1
                    
                if board[i][j] == 1:
                    if count < 2 or count > 3:
                        board[i][j] = 2
                else:
                    if count == 3:
                        board[i][j] = -1
        
        for i in range(row):
            for j in range(col):
                if board[i][j] == -1:
                    board[i][j] = 1
                elif board[i][j] == 2:
                    board[i][j] = 0
```



# [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

难度 困难

序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

**示例:** 

```
你可以将以下二叉树：

    1
   / \
  2   3
     / \
    4   5

序列化为 "[1,2,3,null,null,4,5]"
```

**提示:** 这与 LeetCode 目前使用的方式一致，详情请参阅 [LeetCode 序列化二叉树的格式](https://leetcode-cn.com/faq/#binary-tree)。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

**说明:** 不要使用类的成员 / 全局 / 静态变量来存储状态，你的序列化和反序列化算法应该是无状态的。



**解法**

先序遍历。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        return self.serialize2(root, '')

    def serialize2(self, root, string):
        if root == None:
            string += 'None,'
        else:
            string += (str(root.val) + ',')
            string = self.serialize2(root.left, string)
            string = self.serialize2(root.right, string)
        return string

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        stringList = data.split(',')
        return self.deserialize2(stringList)

    def deserialize2(self, stringList):
        if stringList[0] == 'None':
            del stringList[0]
            return None
        
        val = int(stringList[0])
        root = TreeNode(val)
        del stringList[0]
        root.left = self.deserialize2(stringList)
        root.right = self.deserialize2(stringList)
        return root

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```



# [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

难度 中等

给定一个无序的整数数组，找到其中最长上升子序列的长度。

**示例:**

```
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

**说明:**

- 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
- 你算法的时间复杂度应该为 O(*n^2*) 。



**解法**

+ 方法一：动态规划。时间复杂度：$O(n^2)$ ，空间复杂度：$O(n)$ 。

+ 方法二：贪心 + 二分查找。

  设当前已求出的最长上升子序列的长度为 $\textit{len}$（初始时为 1），从前往后遍历数组 $\textit{nums}$，在遍历到 $\textit{nums}[i]$ 时：

  如果 $\textit{nums}[i] > d[\textit{len}]$ ，则直接加入到 $d$ 数组末尾，并更新 $\textit{len} = \textit{len} + 1$  ；

  否则，在 $d$ 数组中二分查找，找到第一个比 $\textit{nums}[i]$ 小的数 $d[k]$ ，并更新 $d[k + 1] = \textit{nums}[i]$ 。

  时间复杂度：$O(n\log n)$ 。空间复杂度：$O(n)$ 。



**代码**

```python
# 方法一
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0

        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(0, i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[j] + 1, dp[i])
              
        return max(dp)

# 方法二
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        d = []

        for num in nums:
            if not d or num > d[-1]:
                d.append(num)
            else:
                l = 0
                r = len(d) - 1
                loc = r
                while l <= r:
                    mid = (l + r) // 2
                    if d[mid] >= num:
                        loc = mid
                        r = mid - 1  
                    else:
                        l = mid + 1
                d[loc] = num

        return len(d)
```



# [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

难度 中等

给定一个整数数组，其中第 *i* 个元素代表了第 *i* 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

**示例:**

```
输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```



**解法**

动态规划。详见代码。时间复杂度：$O(N)$  空间复杂度： $O(N)$ ， $N$ 为天数。




**代码**

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        f0 = -prices[0]
        f1 = f2 = 0

        # 0 : 目前持有一支股票，对应的「累计最大收益」
        # 1 : 目前不持有任何股票，并且处于冷冻期中，对应的「累计最大收益」
        # 2 : 目前不持有任何股票，并且不处于冷冻期中，对应的「累计最大收益」
        for i in range(1, len(prices)):
            f0, f1, f2 = max(f0, f2 - prices[i]), f0 + prices[i], max(f1, f2)

        return max(f1, f2)
```



# [312. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)

难度 困难

有 `n` 个气球，编号为`0` 到 `n-1`，每个气球上都标有一个数字，这些数字存在数组 `nums` 中。

现在要求你戳破所有的气球。如果你戳破气球 `i` ，就可以获得 `nums[left] * nums[i] * nums[right]` 个硬币。 这里的 `left` 和 `right` 代表和 `i` 相邻的两个气球的序号。注意当你戳破了气球 `i` 后，气球 `left` 和气球 `right` 就变成了相邻的气球。

求所能获得硬币的最大数量。

**说明:**

- 你可以假设 `nums[-1] = nums[n] = 1`，但注意它们不是真实存在的所以并不能被戳破。
- 0 ≤ `n` ≤ 500, 0 ≤ `nums[i]` ≤ 100

**示例:**

```
输入: [3,1,5,8]
输出: 167 
解释: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
     coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
```



**解法**

动态规划。

戳破气球 k ，得先把开区间 (i, k) 的气球都戳破，再把开区间 (k, j) 的气球都戳破；最后剩下的气球 k，相邻的就是气球 i 和气球 j，这时候戳破 k 的话得到的分数就是 points[i] × points[k] × points[j]。戳破开区间 (i, k) 和开区间 (k, j) 的气球最多能得到的分数就是 dp\[i\]\[k\] 和 dp\[k\]\[j\]，这恰好就是对 dp 数组的定义。

时间复杂度： $O(N^3)$ ，空间复杂度： $O(N^2)$ 



**代码**

```python
# 官方题解
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        rec = [[0] * (n + 2) for _ in range(n + 2)]
        val = [1] + nums + [1]

        for i in range(n - 1, -1, -1):
            for j in range(i + 2, n + 2):
                for k in range(i + 1, j):
                    total = val[i] * val[k] * val[j]
                    total += rec[i][k] + rec[k][j]
                    rec[i][j] = max(rec[i][j], total)
        
        return rec[0][n + 1]
```



# [315. 计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)

难度 困难

给定一个整数数组 *nums*，按要求返回一个新数组 *counts*。数组 *counts* 有该性质： `counts[i]` 的值是 `nums[i]` 右侧小于 `nums[i]` 的元素的数量。

 

**示例：**

```
输入：[5,2,6,1]
输出：[2,1,1,0] 
解释：
5 的右侧有 2 个更小的元素 (2 和 1)
2 的右侧仅有 1 个更小的元素 (1)
6 的右侧有 1 个更小的元素 (1)
1 的右侧有 0 个更小的元素
```



**解法**

从尾部开始遍历数组。将遇到过的数字放入排序列表中。遇到新数字时在列表中进行二分查找，即可确定小于该数的数值的数量。时间复杂度： $O(N (\log(N) + N))$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        
        length = len(nums)
        if length == 0:
            return []
        elif length == 1:
            return [0]
        
        self.sortList = [nums[-1]]
        ans = [0 for i in range(length)]

        for i in range(length - 2, -1, -1):
            ans[i] = self.search(nums[i])
        
        return ans 

    def search(self, target):
        left = 0
        right = len(self.sortList) - 1
        find = None

        while left <= right:
            mid = (left + right) // 2
            if self.sortList[mid] < target:
                left = mid + 1
            elif self.sortList[mid] > target:
                right = mid - 1
            else:
                find = mid 
                break

        if find != None:
            while find >= 0 and self.sortList[find] == target:
                find -= 1
            self.sortList.insert(find + 1, target)
            return find + 1 if find >= 0 else 0
        else:
            self.sortList.insert(left, target)
            return right + 1
```



# [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

难度中等510

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

**示例 1:**

```
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
```

**示例 2:**

```
输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
```



**解法**

深度优先搜索 + 动态规划。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root: TreeNode) -> int:

        self.do = collections.defaultdict(lambda :0)
        self.donot = collections.defaultdict(lambda :0)

        def dfs(root):
            if root is None:
                return 

            dfs(root.left)
            dfs(root.right)

            self.do[root] = root.val + self.donot[root.left] + self.donot[root.right]
            self.donot[root] = max(self.do[root.left], self.donot[root.left]) + max(self.do[root.right], self.donot[root.right])
        
        dfs(root)
        return max(self.do[root], self.donot[root])
```



# [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

难度 中等

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

**示例 1:**

```
输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
```

**示例 2:**

```
输入: coins = [2], amount = 3
输出: -1
```

**说明**:
你可以认为每种硬币的数量是无限的。


**解法**

+ 方法一：递归得到所有的可行方案，返回最优解。（时间超时，无法AC）时间复杂度：$O(\text{amount}^n)$ $n$ 为硬币种类，空间复杂度：$O(n)$ 。

+ 方法二：自上而下动态规划。时间复杂度：$O(\text{amount}*n)$ $n$ 为硬币种类，空间复杂度：$O(\text{amount})$ 。

  假设我们知道$F(S)$ ，即组成金额 $S$ 最少的硬币数，最后一枚硬币的面值是 $c$。

  由于问题的最优子结构，转移方程应为：$F(S) = F(S - C) + 1$ ，所以我们需要枚举每个硬币面额值 $c_0, c_1, c_2 \ldots c_{n -1}$ 并选择其中的最小值。下列递推关系成立：$F(S) = \min_{i=0 ... n-1}{ F(S - c_i) } + 1 \ \text{subject to} \ \ S-c_i \geq 0$

+ 方法三：自下而上动态规划。时间复杂度：$O(\text{amount}*n)$ $n$ 为硬币种类，空间复杂度：$O(\text{amount})$ 。

  定义 $F(i)$ 为组成金额 $i$ 所需最少的硬币数量，假设在计算 $F(i)$ 之前，我们已经计算出 $F(0)-F(i-1)$ 的答案。 则 $F(i)$ 对应的转移方程应为：$F(i)=\min_{j=0 \ldots n-1}{F(i -c_j)} + 1$  。


> python functools.lru_cache的作用主要是用来做缓存，能把相对耗时的函数结果进行保存，避免传入相同的参数重复计算。同时，缓存并不会无限增长，不用的缓存会被释放。 


**代码**

``` python
# 方法一
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort()
        res =  self.coinChange2(coins, amount)

        if res:
            return sum(res)
        else:
            return -1
    
    def coinChange2(self, coins, amount):
        t = []
        if len(coins) > 1:
            maxnum = amount // coins[-1]
            for i in range(0, maxnum + 1):
                find = self.coinChange2(coins[:-1], amount - i * coins[-1])
                if find:
                    find.append(i)
                    t.append(find)
            if t:
                r = t[0]
                for tt in t:
                    if sum(tt) < sum(r):
                        r = tt
                return r
            else:
                return None
        else:
            if amount % coins[0] == 0:
                return [amount // coins[0]]
            else:
                return None
            
# 方法二
import functools
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount < 0 or len(coins) == 0:
            return -1
            
        @functools.lru_cache(amount)
        def coinChange2(left):
            if left < 0:
                return -1
            if left == 0:
                return 0
            ways = int(1e9)
            for coin in coins:
                way = coinChange2(left - coin)
                if way >= 0 and way < ways:
                    ways = way + 1
            return ways if ways < int(1e9) else -1

        return coinChange2(amount)
    
# 方法三
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        ways = [int(1e9)] * (amount + 1)
        ways[0] = 0

        for coin in coins:
            for index in range(coin, amount + 1):
                ways[index] = min(ways[index], ways[index - coin] + 1)
        
        return ways[amount] if ways[amount] != int(1e9) else -1
```



# [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

难度 困难

给定一个整数矩阵，找出最长递增路径的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

**示例 1:**

```
输入: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
输出: 4 
解释: 最长递增路径为 [1, 2, 6, 9]。
```

**示例 2:**

```
输入: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
输出: 4 
解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
```

**解法**

深度优先搜索，搜索过程中记录搜索结果。时间复杂度： $O(MN)$ ，空间复杂度： $O(MN)$ 。


**代码**

```python
# 官方代码
class Solution:
    
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        
        @lru_cache(None)
        def dfs(row: int, column: int) -> int:
            best = 1
            for dx, dy in Solution.DIRS:
                newRow, newColumn = row + dx, column + dy
                if 0 <= newRow < rows and 0 <= newColumn < columns and matrix[newRow][newColumn] > matrix[row][column]:
                    best = max(best, dfs(newRow, newColumn) + 1)
            return best

        ans = 0
        rows, columns = len(matrix), len(matrix[0])
        for i in range(rows):
            for j in range(columns):
                ans = max(ans, dfs(i, j))
        return ans
```



# [332. 重新安排行程](https://leetcode-cn.com/problems/reconstruct-itinerary/)

难度 中等

给定一个机票的字符串二维数组 `[from, to]`，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序。所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。

**说明:**

1. 如果存在多种有效的行程，你可以按字符自然排序返回最小的行程组合。例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前
2. 所有的机场都用三个大写字母表示（机场代码）。
3. 假定所有机票至少存在一种合理的行程。

**示例 1:**

```
输入: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
输出: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

**示例 2:**

```
输入: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出: ["JFK","ATL","JFK","SFO","ATL","SFO"]
解释: 另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"]。但是它自然排序更大更靠后。
```



**解法**

深度优先搜索，将搜索结果逆序，得到最终结果。时间复杂度： $O(m \log m)$ ，空间复杂度： $O(m)$ 。其中 $m$ 是边的数量。



**代码**

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        
        def dfs(curr: str):
            while vec[curr]:
                tmp = heapq.heappop(vec[curr])
                dfs(tmp)
            stack.append(curr)

        vec = collections.defaultdict(list)
        for depart, arrive in tickets:
            vec[depart].append(arrive)
        for key in vec:
            heapq.heapify(vec[key])
        
        stack = list()
        dfs("JFK")
        return stack[::-1]
```



# [336. 回文对](https://leetcode-cn.com/problems/palindrome-pairs/)

难度 困难

给定一组 **互不相同** 的单词， 找出所有**不同** 的索引对`(i, j)`，使得列表中的两个单词， `words[i] + words[j]` ，可拼接成回文串。

 

**示例 1：**

```
输入：["abcd","dcba","lls","s","sssll"]
输出：[[0,1],[1,0],[3,2],[2,4]] 
解释：可拼接成的回文串为 ["dcbaabcd","abcddcba","slls","llssssll"]
```

**示例 2：**

```
输入：["bat","tab","cat"]
输出：[[0,1],[1,0]] 
解释：可拼接成的回文串为 ["battab","tabbat"]
```



**解法**

若字符串长度不相等，可将其拆分为两部分，一部分为回文串，另一部分为较短字符串的翻转。时间复杂度： $O(nm^2)$ ，空间复杂度： $O(nm)$ ，其中 $n$ 是字符串的数量，$m$ 是字符串的平均长度。



**代码**

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:

        def findWord(s: str, left: int, right: int) -> int:
            return indices.get(s[left:right+1], -1)
        
        def isPalindrome(s: str, left: int, right: int) -> bool:
            return (sub := s[left:right+1]) == sub[::-1]
        
        n = len(words)
        indices = {word[::-1]: i for i, word in enumerate(words)}
        
        ret = list()
        for i, word in enumerate(words):
            m = len(word)
            for j in range(m + 1):
                if isPalindrome(word, j, m - 1):
                    leftId = findWord(word, 0, j - 1)
                    if leftId != -1 and leftId != i:
                        ret.append([i, leftId])
                if j and isPalindrome(word, 0, j - 1):
                    rightId = findWord(word, j, m - 1)
                    if rightId != -1 and rightId != i:
                        ret.append([rightId, i])

        return ret
```



# [343. 整数拆分](https://leetcode-cn.com/problems/integer-break/)

难度 中等

给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。

**示例 1:**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1。
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
```

**说明:** 你可以假设 *n* 不小于 2 且不大于 58。



**解法**

数学方法。拆成尽可能多的 `3` 。时间复杂度：$O(1)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        
        quotient, remainder = n // 3, n % 3
        if remainder == 0:
            return 3 ** quotient
        elif remainder == 1:
            return 3 ** (quotient - 1) * 4
        else:
            return 3 ** quotient * 2
```



# [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

难度中等

给定一个非空的整数数组，返回其中出现频率前 ***k\*** 高的元素。

 

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

 

**提示：**

- 你可以假设给定的 *k* 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度**必须**优于 O(*n* log *n*) , *n* 是数组的大小。
- 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
- 你可以按任意顺序返回答案。



**解法**

使用堆求解。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        
        # return [num for num, _ in Counter(nums).most_common(k)]

        d = collections.Counter(nums)
        heap, ans = [], []      # 最小堆

        for key in d:
            heapq.heappush(heap, (-d[key], key))

        for _ in range(k):
            ans.append(heapq.heappop(heap)[1])

        return ans
```



# [350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)

难度 简单

给定两个数组，编写一个函数来计算它们的交集。

 

**示例 1：**

```
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
```

**示例 2:**

```
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```

 

**说明：**

- 输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
- 我们可以不考虑输出结果的顺序。

**进阶**：

- 如果给定的数组已经排好序呢？你将如何优化你的算法？
- 如果 *nums1* 的大小比 *nums2* 小很多，哪种方法更优？
- 如果 *nums2* 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？



**解法**

将数组1转换为哈希表，对数组2中每个元素，判断是否在哈希表中。时间复杂度：$O(M + N)$ ，空间复杂度：$O(M)$ ， $M,N$ 分别为两个数组长度。



**代码**

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic = collections.Counter(nums1)

        ans = []
        for i in nums2:
            if i in dic and dic[i] > 0:
                ans.append(i)
                dic[i] -= 1
        
        return ans
```



# [355. 设计推特](https://leetcode-cn.com/problems/design-twitter/)

难度 中等

设计一个简化版的推特(Twitter)，可以让用户实现发送推文，关注/取消关注其他用户，能够看见关注人（包括自己）的最近十条推文。你的设计需要支持以下的几个功能：

1. **postTweet(userId, tweetId)**: 创建一条新的推文
2. **getNewsFeed(userId)**: 检索最近的十条推文。每个推文都必须是由此用户关注的人或者是用户自己发出的。推文必须按照时间顺序由最近的开始排序。
3. **follow(followerId, followeeId)**: 关注一个用户
4. **unfollow(followerId, followeeId)**: 取消关注一个用户

**示例:**

```
Twitter twitter = new Twitter();

// 用户1发送了一条新推文 (用户id = 1, 推文id = 5).
twitter.postTweet(1, 5);

// 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
twitter.getNewsFeed(1);

// 用户1关注了用户2.
twitter.follow(1, 2);

// 用户2发送了一个新推文 (推文id = 6).
twitter.postTweet(2, 6);

// 用户1的获取推文应当返回一个列表，其中包含两个推文，id分别为 -> [6, 5].
// 推文id6应当在推文id5之前，因为它是在5之后发送的.
twitter.getNewsFeed(1);

// 用户1取消关注了用户2.
twitter.unfollow(1, 2);

// 用户1的获取推文应当返回一个列表，其中包含一个id为5的推文.
// 因为用户1已经不再关注用户2.
twitter.getNewsFeed(1);
```



**解法**

使用列表保存已发送的推特，使用字典保存每个推特用户关注的人。

时间复杂度：`postTweet` ： $O(1)$ ，`getNewsFeed` ： $O(MN)$ ，`follow` ： $O(N)$ ，`unfollow` ： $O(N)$ 。$M, N$ 分别为已发推特数和用户总数。

空间复杂度：`postTweet` ： $O(1)$ ，`getNewsFeed` ： $O(N)$ ，`follow` ： $O(1)$ ，`unfollow` ： $O(1)$ 。



**代码**

```python
class Twitter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.message = []
        self.follower = {}
        

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        self.message.append((userId, tweetId))
        

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        count = 0
        res = []
        if userId in self.follower:
            fo = self.follower[userId]
            fo.append(userId)
        else:
            fo = [userId]
        
        for i in range(len(self.message) - 1, -1, -1):
            if self.message[i][0] in fo:
                res.append(self.message[i][1])
                count += 1
                if count == 10:
                    break
        
        return res
        

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId in self.follower:
            if followeeId not in self.follower[followerId]:
                self.follower[followerId].append(followeeId)
        else:
            self.follower[followerId] = [followeeId]
        

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId in self.follower and followeeId in self.follower[followerId]:
            i = self.follower[followerId].index(followeeId)
            del self.follower[followerId][i]



# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
```



# [365. 水壶问题](https://leetcode-cn.com/problems/water-and-jug-problem/)

难度 中等

有两个容量分别为 *x*升 和 *y*升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 *z*升 的水？

如果可以，最后请用以上水壶中的一或两个来盛放取得的 *z升* 水。

你允许：

- 装满任意一个水壶
- 清空任意一个水壶
- 从一个水壶向另外一个水壶倒水，直到装满或者倒空

**示例 1:** (From the famous [*"Die Hard"* example](https://www.youtube.com/watch?v=BVtQNK_ZUJg))

```
输入: x = 3, y = 5, z = 4
输出: True
```

**示例 2:**

```
输入: x = 2, y = 6, z = 5
输出: False
```



**解法**

+ 解法一：深度优先遍历，遍历所有可能的操作。使用一个集合保存已经计算过的操作，防止无限递归。时间复杂度：$O(xy)$，空间复杂度：$O(xy)$ 。
+ 解法二： 裴蜀定理。 有解当且仅当 $z$ 是 $x, y$ 的最大公约数的倍数。 时间复杂度：$O(\log (\min (x,y)))$  计算最大公约数所使用的辗转相除法 ，空间复杂度：$O(1)$ 。



**代码**

```python
# 解法一
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        stack = [(0,0)]
        done = set()

        while stack:
            x_now, y_now = stack.pop()

            if x_now == z or y_now == z or x_now + y_now == z:
                return True
            if (x_now, y_now) in done:
                continue
            
            done.add((x_now, y_now))

            stack.append((x, y_now))
            stack.append((x_now, y))
            stack.append((0, y_now))
            stack.append((x_now, 0))
            stack.append((min(x_now + y_now, x), y_now - min(x - x_now, y_now)))
            stack.append((x_now - min(x_now, y - y_now), min(x_now + y_now, y)))

        return False
    
# 方法二
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        if x + y < z:
            return False
        if x == 0 or y == 0:
            return z == 0 or x + y == z
        return z % math.gcd(x, y) == 0
```



# [392. 判断子序列](https://leetcode-cn.com/problems/is-subsequence/)

难度 简单

给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

**示例 1:**

```
s = "abc", t = "ahbgdc"

返回 true.
```

**示例 2:**

```
s = "axc", t = "ahbgdc"

返回 false.
```

后续挑战 :

如果有大量输入的 S，称作S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？


**解法**

+ 方法一：双指针便利。时间复杂度： $O(M+N)$ ，空间复杂度： $O(1)$ 。 $M,N$ 为字符串 `s,t` 的长度。
+ 方法二：动态规划。对于字符串 `t` ，令 `f[i][j]` 表示从位置 `i` 开始往后，字符 `j` 第一次出现的位置。在遍历字符串 `s` 时，直接通过这些信息进行跳转。时间复杂度： $O(26 \times M+N)$ ，空间复杂度： $O(26 \times M)$ 。 $M,N$ 为字符串 `s,t` 的长度。


**代码**

```python
# 方法一
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        
        index1 = 0
        index2 = 0

        while index1 < len(t) and index2 < len(s):
            if t[index1] == s[index2]:
                index1 += 1
                index2 += 1
            else:
                index1 += 1

        return index2 >= len(s)

# 方法二
# 官方代码
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        n, m = len(s), len(t)
        f = [[0] * 26 for _ in range(m)]
        f.append([m] * 26)

        for i in range(m - 1, -1, -1):
            for j in range(26):
                f[i][j] = i if ord(t[i]) == j + ord('a') else f[i + 1][j]
        
        add = 0
        for i in range(n):
            if f[add][ord(s[i]) - ord('a')] == m:
                return False
            add = f[add][ord(s[i]) - ord('a')] + 1
        
        return True
```

# [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

难度 中等

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 *encoded_string* 正好重复 *k* 次。注意 *k* 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 *k* ，例如不会出现像 `3a` 或 `2[4]` 的输入。

**示例:**

```
s = "3[a]2[bc]", 返回 "aaabcbc".
s = "3[a2[c]]", 返回 "accaccacc".
s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
```



**解法**

扫描字符串，根据数字对重复部分进行复制和拼接。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def decodeString(self, s: str) -> str:

        def copyStr(count, left, right):
            l = s[ : left] if left > 0 else ''
            r = s[right + 1 : ] if right + 1 < len(s) else ''

            return l + count * s[left + 1 : right] + r


        brackets = 0
        bracket = False
        count = 0
        i = 0

        while i < len(s):

            if not bracket and s[i] >= '0' and s[i] <= '9':
                count = count * 10 + int(s[i])
                s = s[0 : i] + s[i + 1 :]
                continue

            elif s[i] == '[':
                if not bracket:
                    bracket = True
                    left = i
                brackets += 1

            elif s[i] == ']':
                brackets -= 1
                if brackets == 0:
                    s = copyStr(count, left, i)
                    count = 0
                    bracket = False
                    i = left
                    continue

            i += 1
        
        return s
```



# [404. 左叶子之和](https://leetcode-cn.com/problems/sum-of-left-leaves/)

难度 简单

计算给定二叉树的所有左叶子之和。

**示例：**

```
    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24
```



**解法**

深度优先搜索。标记节点是否为左子节点，若是左子节点且为叶节点，则计算答案。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:

        ans = 0

        def dfs(root, isLeft):
            if root is None:
                return 

            if isLeft and root.left is None and root.right is None:
                nonlocal ans
                ans += root.val
            
            dfs(root.left, True)
            dfs(root.right, False)
        
        dfs(root, False)
        return ans
```



# [409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/)

难度 简单

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 `"Aa"` 不能当做一个回文字符串。

**注意:**
假设字符串的长度不会超过 1010。

**示例 1:**

```
输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
```



**解法**

+ 方法一：将字符串转换为列表，对列表进行排序，若列表中相邻的两位相同，则可以组成回文串的一部分。时间复杂度：$O(n \log n)$，空间复杂度：$O(n)$ 。
+ 方法二：统计字符串中每个字符出现的次数，每出现两次可以组成回文串的一部分。时间复杂度：$O(n)$，空间复杂度：$O(m)$  $m$ 为字符串中出现的字符的种类。



**代码**

```python
# 方法一
class Solution:
    def longestPalindrome(self, s: str) -> int:
        l = list(s)
        l.sort()
        index = 0
        res = 0
        while index < len(l):
            if index + 1 < len(l) and l[index] == l[index + 1]:
                res += 2
                index += 2
            else:
                index += 1
        if res < len(l):
            res += 1
        return res

# 方法二
class Solution:
    def longestPalindrome(self, s):
        ans = 0
        count = collections.Counter(s)
        for v in count.values():
            ans += v // 2 * 2
            if ans % 2 == 0 and v % 2 == 1:
                ans += 1
        return ans
```


# [410. 分割数组的最大值](https://leetcode-cn.com/problems/split-array-largest-sum/)

难度 困难

给定一个非负整数数组和一个整数 m，你需要将这个数组分成 m 个非空的连续子数组。设计一个算法使得这 m 个子数组各自和的最大值最小。

**注意:**
数组长度 n 满足以下条件:

+ 1 ≤ n ≤ 1000
+ 1 ≤ m ≤ min(50, n)

**示例:**

```
输入:
nums = [7,2,5,10,8]
m = 2

输出:
18

解释:
一共有四种方法将nums分割为2个子数组。
其中最好的方式是将其分为[7,2,5] 和 [10,8]，
因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。
通过次数14,436提交次数29,219
```

**解法**

二分查找 + 贪心。以数组和为最大值，数组中最大值为最小值，进行二分查找。对每个找到的值进行验证，看是否可以满足题意。时间复杂度： $O(n × \log (sum−maxn))$ ，空间复杂度： $O(1))$ 。


**代码**
```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:

        def check(x: int) -> bool:
            total, cnt = 0, 1

            for num in nums:
                if total + num > x:
                    cnt += 1
                    total = num
                else:
                    total += num

            return cnt <= m

        left = max(nums)
        right = sum(nums)

        while left < right:
            mid = (left + right) // 2

            if check(mid):
                right = mid
            else:
                left = mid + 1

        return left
```



# [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

难度 简单

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和。

 

**提示：**

1. `num1` 和`num2` 的长度都小于 5100
2. `num1` 和`num2` 都只包含数字 `0-9`
3. `num1` 和`num2` 都不包含任何前导零
4. **你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式**



**代码**

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:

        if len(num1) < len(num2):
            num1, num2 = num2, num1
        
        index1 = len(num1) - 1
        index2 = len(num2) - 1
        carry = 0
        ans = ''

        while index1 >= 0:
            if index2 >= 0:
                r = ord(num1[index1]) + ord(num2[index2]) + carry - 2 * ord('0')
                index2 -= 1
            else:
                r = ord(num1[index1]) + carry - ord('0')

            carry = r // 10
            r = r % 10
            ans += str(r)
            index1 -= 1
            
        if carry == 1:
            ans += '1'
            
        return ans[::-1]
```



# [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

难度 中等

给你两个 **非空** 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

你可以假设除了数字 0 之外，这两个数字都不会以零开头。

 

**进阶：**

如果输入链表不能修改该如何处理？换句话说，你不能对列表中的节点进行翻转。

 

**示例：**

```
输入：(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 8 -> 0 -> 7
```



**解法**

将两个链表表示的值存入变量，相加得到结构，并构建新链表。时间复杂度：$O(MN)$，空间复杂度：$O(\max(M,N))$ 。 $M, N$ 分别为两个链表的长度。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        add1 = 0
        add2 = 0

        point = l1
        while point:
            add1 = add1 * 10 + point.val
            point = point.next

        point = l2
        while point:
            add2 = add2 * 10 + point.val
            point = point.next

        res = add1 + add2
        head = ListNode(0)
        while True:
            node = ListNode(res % 10)
            res //= 10
            node.next = head.next
            head.next = node
            if res == 0:
                break
        
        return head.next
```



# [459. 重复的子字符串](https://leetcode-cn.com/problems/repeated-substring-pattern/)

难度 简单

给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。

**示例 1:**

```
输入: "abab"

输出: True

解释: 可由子字符串 "ab" 重复两次构成。
```

**示例 2:**

```
输入: "aba"

输出: False
```

**示例 3:**

```
输入: "abcabcabcabc"

输出: True

解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
```



**解法**

将字符串复制并拼接到字符串尾部，去掉拼接得到的字符串的头尾，如果能在新字符串还能找到原来的字符串，则说明满足题目要求。时间复杂度： $O(n^2)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        def kmp(query: str, pattern: str) -> bool:
            n, m = len(query), len(pattern)
            fail = [-1] * m
            for i in range(1, m):
                j = fail[i - 1]
                while j != -1 and pattern[j + 1] != pattern[i]:
                    j = fail[j]
                if pattern[j + 1] == pattern[i]:
                    fail[i] = j + 1
            match = -1
            for i in range(1, n - 1):
                while match != -1 and pattern[match + 1] != query[i]:
                    match = fail[match]
                if pattern[match + 1] == query[i]:
                    match += 1
                    if match == m - 1:
                        return True
            return False
        
        return kmp(s + s, s)
        # return (s + s).find(s, 1) != len(s)
```



# [460. LFU缓存](https://leetcode-cn.com/problems/lfu-cache/)

难度 困难 

设计并实现[最不经常使用（LFU）](https://baike.baidu.com/item/缓存算法)缓存的数据结构。它应该支持以下操作：`get` 和 `put`。

`get(key)` - 如果键存在于缓存中，则获取键的值（总是正数），否则返回 -1。
`put(key, value)` - 如果键不存在，请设置或插入值。当缓存达到其容量时，它应该在插入新项目之前，使最不经常使用的项目无效。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，**最近**最少使用的键将被去除。

**进阶：**
你是否可以在 **O(1)** 时间复杂度内执行两项操作？

**示例：**

```
LFUCache cache = new LFUCache( 2 /* capacity (缓存容量) */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // 返回 1
cache.put(3, 3);    // 去除 key 2
cache.get(2);       // 返回 -1 (未找到key 2)
cache.get(3);       // 返回 3
cache.put(4, 4);    // 去除 key 1
cache.get(1);       // 返回 -1 (未找到 key 1)
cache.get(3);       // 返回 3
cache.get(4);       // 返回 4
```



**解法**

使用双哈希表。一个保存项目地址，一个频率相同的项目。 时间复杂度：`get` 时间复杂度 $O(1)$， `put` 时间复杂度 $O(1)$ 。  空间复杂度：$O(\textit{capacity})$ 。



**代码**

```python
# 官方题解
class Node:
    def __init__(self, key, val, pre=None, nex=None, freq=0):
        self.pre = pre
        self.nex = nex
        self.freq = freq
        self.val = val
        self.key = key
        
    def insert(self, nex):
        nex.pre = self
        nex.nex = self.nex
        self.nex.pre = nex
        self.nex = nex
    
def create_linked_list():
    head = Node(0, 0)
    tail = Node(0, 0)
    head.nex = tail
    tail.pre = head
    return (head, tail)

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.minFreq = 0
        self.freqMap = collections.defaultdict(create_linked_list)
        self.keyMap = {}

    def delete(self, node):
        if node.pre:
            node.pre.nex = node.nex
            node.nex.pre = node.pre
            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
                self.freqMap.pop(node.freq)
        return node.key
        
    def increase(self, node):
        node.freq += 1
        self.delete(node)
        self.freqMap[node.freq][-1].pre.insert(node)
        if node.freq == 1:
            self.minFreq = 1
        elif self.minFreq == node.freq - 1:
            head, tail = self.freqMap[node.freq - 1]
            if head.nex is tail:
                self.minFreq = node.freq

    def get(self, key: int) -> int:
        if key in self.keyMap:
            self.increase(self.keyMap[key])
            return self.keyMap[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity != 0:
            if key in self.keyMap:
                node = self.keyMap[key]
                node.val = value
            else:
                node = Node(key, value)
                self.keyMap[key] = node
                self.size += 1
            if self.size > self.capacity:
                self.size -= 1
                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
                self.keyMap.pop(deleted)
            self.increase(node)
            
# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



# [466. 统计重复个数](https://leetcode-cn.com/problems/count-the-repetitions/)

难度 困难

由 n 个连接的字符串 s 组成字符串 S，记作 `S = [s,n]`。例如，`["abc",3]`=“abcabcabc”。

如果我们可以从 s2 中删除某些字符使其变为 s1，则称字符串 s1 可以从字符串 s2 获得。例如，根据定义，"abc" 可以从 “abdbec” 获得，但不能从 “acbbe” 获得。

现在给你两个非空字符串 s1 和 s2（每个最多 100 个字符长）和两个整数 0 ≤ n1 ≤ 106 和 1 ≤ n2 ≤ 106。现在考虑字符串 S1 和 S2，其中 `S1=[s1,n1]` 、`S2=[s2,n2]` 。

请你找出一个可以满足使`[S2,M]` 从 `S1` 获得的最大整数 M 。

 

**示例：**

```
输入：
s1 ="acb",n1 = 4
s2 ="ab",n2 = 2

返回：
2
```



**思路**

寻找循环节。 时间复杂度：$O(|s1|×|s2|)$ ， 空间复杂度：$O(|s2|)$  。 



**代码**

```python
# 官方题解
class Solution:
    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        if n1 == 0:
            return 0
        
        s1cnt, index, s2cnt = 0, 0, 0
        recall = {}

        # 如果我们之前遍历了 s1cnt 个 s1 时，匹配到的是第 s2cnt 个 s2 中同样的第 index 个字符，那么就有循环节了

        while True:
            # 我们多遍历一个 s1，看看能不能找到循环节
            s1cnt += 1
            for ch in s1:
                if ch == s2[index]:
                    index += 1
                    if index == len(s2):
                        s2cnt, index = s2cnt + 1, 0
            # 还没有找到循环节，所有的 s1 就用完了
            if s1cnt == n1:
                return s2cnt // n2
            # 出现了之前的 index，表示找到了循环节
            if index in recall:
                s1cnt_prime, s2cnt_prime = recall[index]
                # 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
                pre_loop = (s1cnt_prime, s2cnt_prime)
                # 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
                in_loop = (s1cnt - s1cnt_prime, s2cnt - s2cnt_prime)
                break
            else:
                recall[index] = (s1cnt, s2cnt)

        # ans 存储的是 S1 包含的 s2 的数量，考虑的之前的 pre_loop 和 in_loop
        ans = pre_loop[1] + (n1 - pre_loop[0]) // in_loop[0] * in_loop[1]
        # S1 的末尾还剩下一些 s1，我们暴力进行匹配
        rest = (n1 - pre_loop[0]) % in_loop[0]
        for i in range(rest):
            for ch in s1:
                if ch == s2[index]:
                    index += 1
                    if index == len(s2):
                        ans, index = ans + 1, 0
        # S1 包含 ans 个 s2，那么就包含 ans / n2 个 S2
        return ans // n2
```



# [486. 预测赢家](https://leetcode-cn.com/problems/predict-the-winner/)

难度中等258

给定一个表示分数的非负整数数组。 玩家 1 从数组任意一端拿取一个分数，随后玩家 2 继续从剩余数组任意一端拿取分数，然后玩家 1 拿，…… 。每次一个玩家只能拿取一个分数，分数被拿取之后不再可取。直到没有剩余分数可取时游戏结束。最终获得分数总和最多的玩家获胜。

给定一个表示分数的数组，预测玩家1是否会成为赢家。你可以假设每个玩家的玩法都会使他的分数最大化。

 

**示例 1：**

```
输入：[1, 5, 2]
输出：False
解释：一开始，玩家1可以从1和2中进行选择。
如果他选择 2（或者 1 ），那么玩家 2 可以从 1（或者 2 ）和 5 中进行选择。如果玩家 2 选择了 5 ，那么玩家 1 则只剩下 1（或者 2 ）可选。
所以，玩家 1 的最终分数为 1 + 2 = 3，而玩家 2 为 5 。
因此，玩家 1 永远不会成为赢家，返回 False 。
```

**示例 2：**

```
输入：[1, 5, 233, 7]
输出：True
解释：玩家 1 一开始选择 1 。然后玩家 2 必须从 5 和 7 中进行选择。无论玩家 2 选择了哪个，玩家 1 都可以选择 233 。
     最终，玩家 1（234 分）比玩家 2（12 分）获得更多的分数，所以返回 True，表示玩家 1 可以成为赢家。
```

 

**提示：**

- 1 <= 给定的数组长度 <= 20.
- 数组里所有分数都为非负数且不会大于 10000000 。
- 如果最终两个玩家的分数相等，那么玩家 1 仍为赢家。



**解法**

动态规划。`dp[i][j]` 表示当数组剩下的部分为下标 `i` 到下标 `j` 时，当前玩家与另一个玩家的分数之差的最大值，注意当前玩家不一定是先手。当 `i=j` 时，只剩一个数字，当前玩家只能拿取这个数字。状态转移方程：
$$
dp[i][j]= \max(nums[i]−dp[i+1][j],nums[j]−dp[i][j−1])
$$
 时间复杂度： $O(n^2)$ ，空间复杂度：$O(n)$ 。



**代码**

```python
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        length = len(nums)
        dp = [0] * length
        for i, num in enumerate(nums):
            dp[i] = num
        for i in range(length - 2, -1, -1):
            for j in range(i + 1, length):
                dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])
        return dp[length - 1] >= 0
```



# [491. 递增子序列](https://leetcode-cn.com/problems/increasing-subsequences/)

难度 中等

给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。

**示例:**

```
输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
```

**说明:**

1. 给定数组的长度不会超过15。
2. 数组中的整数范围是 [-100,100]。
3. 给定数组中可能包含重复数字，相等的数字应该被视为递增的一种情况。



**解法**

枚举 + 减枝。时间复杂度： $O(2^n n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        import copy
        
        ans = []
        temp = []

        def dfs(index, last):
            if index == len(nums):
                if len(temp) >= 2:
                    ans.append(copy.copy(temp))
                return 
            
            if nums[index] >= last:
                temp.append(nums[index])
                dfs(index + 1, nums[index])
                temp.pop()
            
            if nums[index] != last:
                dfs(index + 1, last)
        
        dfs(0, -101)
        return ans
```

