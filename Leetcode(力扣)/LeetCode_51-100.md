[toc]



# [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

难度 简单

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例:**

```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**进阶:**

如果你已经实现复杂度为 O(*n*) 的解法，尝试使用更为精妙的分治法求解。



**解法**

动态规划：$f(i) = \max\{f(i-1)+a_i, a_i\}$ 。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = nums[0]
        pre = 0

        for num in nums:
            pre = max(pre + num, num)
            ans = max(ans, pre)
        
        return ans
```

# [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

难度 中等

给定一个包含 *m* x *n* 个元素的矩阵（*m* 行, *n* 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

**示例 1:**

```
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]
```

**示例 2:**

```
输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]
```



**解法**

使用指针保存当前打印矩阵的上下左右边界。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []

        left = 0
        right = len(matrix[0]) - 1
        top = 0
        bottom = len(matrix) - 1

        ans = []

        while right >= left:
            for i in range(left, right + 1):
                ans.append(matrix[top][i])
            
            if top == bottom:
                break

            for j in range(top + 1, bottom + 1):
                ans.append(matrix[j][right])

            if left == right:
                break
                
            for i in range(right - 1, left - 1, -1):
                ans.append(matrix[bottom][i])
            
            if top + 1 == bottom:
                break

            for j in range(bottom - 1, top, -1):
                ans.append(matrix[j][left])

            left += 1
            right -= 1
            top += 1
            bottom -= 1
        
        return ans

```



# [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

难度 中等

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

**示例 1:**

```
输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
```

**示例 2:**

```
输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
```



**解法**

维护一个变量指向目前可以到达的最远位置，遍历数组，更新该变量。时间复杂度：$O(N)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        far = 0                         # 最远可到的距离
        for i in range(len(nums)):
            if i + nums[i] > far:
                far = i + nums[i]
            if far >= len(nums) - 1:
                return True
            if far == i:                # 卡在某处
                break

        return False
```


# [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

难度 中等 

给出一个区间的集合，请合并所有重叠的区间。

**示例 1:**

```
输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

**示例 2:**

```
输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
```



**解法**

先排序，再合并。时间复杂度：$O(N\log N)$ ，空间复杂度：$O(\log N)$ （排序所需空间）。



**代码**

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x: x[0])

        res = []
        index = 0
        for inte in intervals:
            if not res or res[-1][1] < inte[0]:
                res.append(inte)
            else:
                res[-1][1] = max(res[-1][1], inte[1])

        return res
```



# [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

难度 中等

一个机器人位于一个 *m x n* 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png)

网格中的障碍物和空位置分别用 `1` 和 `0` 来表示。

**说明：***m* 和 *n* 的值均不超过 100。

**示例 1:**

```
输入:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
输出: 2
解释:
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
```



**解法**

动态规划。使用滚动数组进行优化。时间复杂度： $O(N)$ ，空间复杂度： $O(MN)$ 。



**代码**

```python
import queue
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        ans = [0] * n
        
        ans[0] = 1 if obstacleGrid[0][0] == 0 else 0

        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    ans[j] = 0
                elif j - 1 >= 0 and obstacleGrid[i][j-1] == 0:
                    ans[j] += ans[j - 1]

        return ans[-1]
```



# [67. 二进制求和](https://leetcode-cn.com/problems/add-binary/)

难度 简单

给你两个二进制字符串，返回它们的和（用二进制表示）。

输入为 **非空** 字符串且只包含数字 `1` 和 `0`。

 

**示例 1:**

```
输入: a = "11", b = "1"
输出: "100"
```

**示例 2:**

```
输入: a = "1010", b = "1011"
输出: "10101"
```

 

**提示：**

- 每个字符串仅由字符 `'0'` 或 `'1'` 组成。
- `1 <= a.length, b.length <= 10^4`
- 字符串如果不是 `"0"` ，就都不含前导零。





**解法**

+ 方法一：先转十进制，计算得到结果再转二进制。
+ 方法二：模拟加法过程。时间复杂度： $O(\max(M,N))$ ， 空间复杂度：  $O(\max(M,N))$ 。



**代码**

```python
# 方法一
class Solution:
    def addBinary(self, a, b) -> str:
        return '{0:b}'.format(int(a, 2) + int(b, 2))
    	# return bin(int(a, 2) + int(b, 2))[2:]

# 方法二
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        if len(b) > len(a):
            a, b = b, a 

        i = len(a) - 1
        j = len(b) - 1
        ans = ''
        carry = 0

        while i >= 0 and j >= 0:
            if a[i] == '0' and b[j] == '0':
                if carry == 0:
                    ans = '0' + ans
                else:
                    ans = '1' + ans
                    carry = 0
            elif a[i] == '1' and b[j] == '1':
                if carry == 0:
                    ans = '0' + ans
                    carry = 1
                else:
                    ans = '1' + ans
            else:
                if carry == 0:
                    ans = '1' + ans
                else:
                    ans = '0' + ans
            i -= 1
            j -= 1
        
        while i >= 0:
            if carry == 1:
                if a[i] == '1':
                    ans = '0' + ans
                else:
                    ans = '1' + ans
                    carry = 0
            else:
                ans = a[i] + ans
            i -= 1
        
        if carry == 1:
            ans = '1' + ans

        return ans
```



# [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

难度 简单

实现 `int sqrt(int x)` 函数。

计算并返回 *x* 的平方根，其中 *x* 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

**示例 1:**

```
输入: 4
输出: 2
```

**示例 2:**

```
输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```



**解法**

方法一：`sqrt` 函数。

方法二：牛顿迭代法。令 $y=x^2-C$ 。任取一个 $x_0$ 作为初始值，在每一步的迭代中，我们找到函数图像上的点 $(x_i, f(x_i))$ 过该点作一条斜率为该点导数 $f'(x_i)$ 的直线，与横轴的交点记为 $x_{i+1}$ 。 $x_{i+1}$ 相较于 $x_i$ 而言距离零点更近。在经过多次迭代后，我们就可以得到一个距离零点非常接近的交点。时间复杂度： $O(\log N)$ 。空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def mySqrt(self, x: int) -> int:
        return int(math.sqrt(x))
    
# 方法二 官方题解
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        
        C, x0 = float(x), float(x)
        while True:
            xi = 0.5 * (x0 + C / x0)
            if abs(x0 - xi) < 1e-7:
                break
            x0 = xi
        
        return int(x0)
```



# [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

难度 简单

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**注意：**给定 *n* 是一个正整数。

**示例 1：**

```
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
```

**示例 2：**

```
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶
```



**解法**

+ 方法一：递归。时间复杂度：$O(2^N)$， 空间复杂度： $O(2^N)$ 。（使用 `@lru_cache` ： 时间复杂度：$O(N)$， 空间复杂度： $O(N)$ 。）
+ 方法二：动态规划。时间复杂度：$O(N)$， 空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    @lru_cache
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            return self.climbStairs(n - 1) + self.climbStairs(n - 2)

# 方法二
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [2, 1]

        if n == 1:
            return 1
        else:
            for i in range(n - 3, -1, -1):
                dp[0], dp[1] = dp[0] + dp[1], dp[0]
            return dp[0] 
```




# [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

难度 困难

给你两个单词 *word1* 和 *word2*，请你计算出将 *word1* 转换成 *word2* 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

1. 插入一个字符
2. 删除一个字符
3. 替换一个字符

 

**示例 1：**

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

**示例 2：**

```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```



**解法**

动态规划。

操作可简化为三种情况：

1. 在 word1 末尾加一字符
2. 在 word2 末尾加一字符
3. 替换 word1 中一个字符

 用 `D[i][j]` 表示 `A` 的前 `i` 个字母和 `B` 的前 `j` 个字母之间的编辑距离。 得到转移方程：

那么我们可以写出如下的状态转移方程：

+ 若 A 和 B 的最后一个字母相同：

$$
\begin{aligned} D[i][j] &= \min(D[i][j - 1] + 1, D[i - 1][j]+1, D[i - 1][j - 1])\\ &= 1 + \min(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1] - 1) \end{aligned}
$$

+ 若 A 和 B 的最后一个字母不同：

$$
D[i][j] = 1 + \min(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1])
$$

时间复杂度：$O(MN)$ ，空间复杂度：$O(MN)$  。 $M N$ 分别为 word1 word2 长度。



**代码**

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)

        # 有空串
        if len1 == 0 or len2 == 0:
            return len1 + len2
        
        dp = [[0] * (len2 + 1) for i in range(len1 + 1)]

        # 初始化边界
        for i in range(len1 + 1):
            dp[i][0] = i 
        for i in range(len2 + 1):
            dp[0][i] = i

        # 计算所有 dp 值
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # word1 末尾加一个字符
                    dp[i][j - 1] + 1,		# word2 末尾加一个字符
                    dp[i - 1][j - 1] + 1 if word1[i - 1] != word2[j - 1] else dp[i - 1][j - 1]      
                    					# 修改一个字符
                )

        return dp[len1][len2]
```



# [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

难度 困难

给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字符的最小子串。

**示例：**

```
输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
```

**说明：**

- 如果 S 中不存这样的子串，则返回空字符串 `""`。
- 如果 S 中存在这样的子串，我们保证它是唯一的答案。



**解法**

滑动窗口。时间复杂度：$O(C⋅∣s∣+∣t∣)$  ，空间复杂度：$O(C)$ ，$C$为字符集大小。



**代码**

```cpp
class Solution {
public:
    unordered_map <char, int> ori, cnt;

    bool check() {
        for (const auto &p: ori) {
            if (cnt[p.first] < p.second) {
                return false;
            }
        }
        return true;
    }

    string minWindow(string s, string t) {
        for (const auto &c: t) {
            ++ori[c];
        }

        int l = 0, r = -1;
        int len = INT_MAX, ansL = -1, ansR = -1;

        while (r < int(s.size())) {
            if (ori.find(s[++r]) != ori.end()) {
                ++cnt[s[r]];
            }
            while (check() && l <= r) {
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                }
                if (ori.find(s[l]) != ori.end()) {
                    --cnt[s[l]];
                }
                ++l;
            }
        }

        return ansL == -1 ? string() : s.substr(ansL, len);
    }
};
```



# [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

难度 困难

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

 

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png)

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

 

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

 

**示例:**

```
输入: [2,1,5,6,2,3]
输出: 10
```



**解法**

使用栈保存边界信息。 例如 `[6, 7, 5, 2, 4, 5, 9, 3]` ，对于左边界栈结果为： ` [2(3), 3(7)] ` ，从而得到各点的左边界为： `[-1, 0, -1, -1, 3, 4, 5, 3]` 。同理可得到各点的右边界。根据左右边界和高度可以计算得到面积。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        left, right = [0] * n, [0] * n

        mono_stack = list()
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)
        
        mono_stack = list()
        for i in range(n - 1, -1, -1):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            right[i] = mono_stack[-1] if mono_stack else n
            mono_stack.append(i)
        
        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n)) if n > 0 else 0
        return ans
```



# [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

难度 困难

给定三个字符串 *s1*, *s2*, *s3*, 验证 *s3* 是否是由 *s1* 和 *s2* 交错组成的。

**示例 1:**

```
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出: true
```

**示例 2:**

```
输入: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
输出: false
```



**解法**

动态规划，并使用滚动字符串进行优化。时间复杂度： $O(MN)$ ，空间复杂度： $O(N)$ 。 $M,N$ 分别为字符串 `s1,s2` 的长度



**代码**

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:

        if len(s1) + len(s2) != len(s3):
            return False

        dp = [False for _ in range(len(s2) + 1)]
        dp[0] = True

        for i in range(0, len(s1) + 1):
            for j in range(0, len(s2) + 1):
                index = i + j - 1

                if i > 0:
                    dp[j] &= (s3[index] == s1[i - 1])
                
                if j > 0:
                    dp[j] |= (dp[j - 1] and s3[index] == s2[j - 1]) 

        return dp[-1]
```



# [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

难度 中等

给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：

- 节点的左子树只包含**小于**当前节点的数。
- 节点的右子树只包含**大于**当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**示例 1:**

```
输入:
    2
   / \
  1   3
输出: true
```

**示例 2:**

```
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```



**解法**

进行中序遍历，验证遍历结果是否为递增序列。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True

        tree =[]

        # 中序遍历
        def search(node):
            if node.left == None and node.right == None:
                tree.append(node.val)
            else:
                if node.left != None:
                    search(node.left)
                
                tree.append(node.val)

                if node.right != None:
                    search(node.right)
        
        search(root)

        ans = True
        for i in range(len(tree) - 1):
            if tree[i] >= tree[i + 1]:
                ans = False
                break

        return ans
```

