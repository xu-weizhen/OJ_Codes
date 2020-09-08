[toc]



# [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

难度 困难

*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/8-queens.png)

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回所有不同的 *n* 皇后问题的解决方案。

每一种解法包含一个明确的 *n* 皇后问题的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

 

**示例：**

```
输入：4
输出：[
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
```

 

**提示：**

- 皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。



**解法**

递归。使用集合表示当前不可用的列和对角线。时间复杂度： $O(N!)$ ，空间复杂度： $O(n)$ 。使用位替代集合，可将时间复杂度降为 $O(1)$ 。



**代码**

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:

        ans = []
        now = []
        col = set()
        dia1 = set()
        dia2 = set()

        def vis():
            a = []
            for num in now:
                r = ['.'] * n 
                r[num] = 'Q'
                r = ''.join(r)
                a.append(r)
            ans.append(a)


        def backtrack(row):
            if row == n:
                vis()
            else:
                for i in range(n):
                    if i in col or i - row in dia1 or i + row in dia2:
                        continue 
                    now.append(i)

                    col.add(i)
                    dia1.add(i - row)
                    dia2.add(i + row)

                    backtrack(row + 1)

                    col.remove(i)
                    dia1.remove(i - row)
                    dia2.remove(i + row)
                    now.pop()
        
        backtrack(0)
        return ans
```



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



# [60. 第k个排列](https://leetcode-cn.com/problems/permutation-sequence/)

难度 中等

给出集合 `[1,2,3,…,*n*]`，其所有元素共有 *n*! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 *n* = 3 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 *n* 和 *k*，返回第 *k* 个排列。

**说明：**

- 给定 *n* 的范围是 [1, 9]。
- 给定 *k* 的范围是[1,  *n*!]。

**示例 1:**

```
输入: n = 3, k = 3
输出: "213"
```

**示例 2:**

```
输入: n = 4, k = 9
输出: "2314"
```



**解法**

先列出不同长度的字符串有几种排列方案，从而根据 `k` 值确定每一位的数字。时间复杂度： $O(n^2)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
# 官方题解
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        factorial = [1]
        for i in range(1, n):
            factorial.append(factorial[-1] * i)
        
        k -= 1
        ans = list()
        valid = [1] * (n + 1)	# 记录该数字是否已经被用过
        for i in range(1, n + 1):
            order = k // factorial[n - i] + 1
            for j in range(1, n + 1):
                order -= valid[j]
                if order == 0:
                    ans.append(str(j))
                    valid[j] = 0
                    break
            k %= factorial[n - i]

        return "".join(ans)
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



# 64. 最小路径和

给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明**：每次只能向下或者向右移动一步。

**示例:**

```
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```


**解法**

动态规划 + 滚动数组优化。时间复杂度： $O(M,N)$ ， 空间复杂度：  $O(N)$ ， $M,N$ 分别为行数和列数。


**代码**

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:

        if not grid or not grid[0]:
            return 0
            
        m = len(grid)
        n = len(grid[0])

        dp = [0 for _ in range(n)]
        dp[0]= grid[0][0]
        for i in range(1, n):
            dp[i] = dp[i - 1] + grid[0][i]
        
        for i in range(1, m):
            dp[0] += grid[i][0]

            for j in range(1, n):
                dp[j] = min(dp[j - 1], dp[j]) + grid[i][j]
        
        return dp[-1]
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



# [77. 组合](https://leetcode-cn.com/problems/combinations/)

难度 中等

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

**示例:**

```
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```



**解法**

递归。时间复杂度： $O(n!)$ ，空间复杂度： $O(k^2)$ 。



**代码**

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # return list(itertools.combinations(range(1,n+1),k))
        
        import copy

        ans = []

        def dfs(index, k):
            tans.append(index)

            if k == 1:
                ans.append(copy.copy(tans))
                tans.pop()
                return 
            
            nonlocal n 
            
            for i in range(index + 1, n + 1):
                dfs(i, k - 1)
            
            tans.pop()
        
        for i in range(1, n + 1):
            tans = []
            dfs(i, k)

        return ans 
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



# [93. 复原IP地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

难度中等336

给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。

有效的 IP 地址正好由四个整数（每个整数位于 0 到 255 之间组成），整数之间用 `'.' `分隔。

 

**示例:**

```
输入: "25525511135"
输出: ["255.255.11.135", "255.255.111.35"]
```



**解法**

+ 方法一：直接 `for ` 循环。时间复杂度： $O(1)$ ，空间复杂度： $O(1)$ 。
+ 方法二：递归。时间复杂度： $O(|s|)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result = []

        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(1, 4):
                    for d in range(1, 4):
                        if a + b + c + d == len(s):
                            if self.isLegal(s[:a], s[a:a + b], s[a + b : a + b + c], s[a + b + c:]):
                                result.append(s[:a] + '.' + s[a:a + b] + '.' + s[a + b: a + b + c] + '.' + s[a + b + c:])  
        return result 
    
    def isLegal(self, a, b, c, d):
        r = [a, b, c, d]

        for addr in r:
            if len(addr) > 1 and addr[0] == '0':
                return False
            if int(addr) > 255:
                return False 
        
        return True
    
# 方法二
# 官方代码
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        SEG_COUNT = 4
        ans = list()
        segments = [0] * SEG_COUNT
        
        def dfs(segId: int, segStart: int):
            # 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
            if segId == SEG_COUNT:
                if segStart == len(s):
                    ipAddr = ".".join(str(seg) for seg in segments)
                    ans.append(ipAddr)
                return
            
            # 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
            if segStart == len(s):
                return

            # 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
            if s[segStart] == "0":
                segments[segId] = 0
                dfs(segId + 1, segStart + 1)
            
            # 一般情况，枚举每一种可能性并递归
            addr = 0
            for segEnd in range(segStart, len(s)):
                addr = addr * 10 + (ord(s[segEnd]) - ord("0"))
                if 0 < addr <= 0xFF:
                    segments[segId] = addr
                    dfs(segId + 1, segEnd + 1)
                else:
                    break
        

        dfs(0, 0)
        return ans
```





# [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

难度 中等

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的 **二叉搜索树** 。

 

**示例：**

```
输入：3
输出：
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释：
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
```

 

**提示：**

- `0 <= n <= 8`



**解法**

递归。时间复杂度： $O(\frac{4^N}{\sqrt N})$ ，空间复杂度： $O(\frac{4^N}{\sqrt N})$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def generateTrees(start, end):
            if start > end:
                return [None,]
            
            allTrees = []
            for i in range(start, end + 1):  # 枚举可行根节点
                # 获得所有可行的左子树集合
                leftTrees = generateTrees(start, i - 1)
                
                # 获得所有可行的右子树集合
                rightTrees = generateTrees(i + 1, end)
                
                # 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
                for l in leftTrees:
                    for r in rightTrees:
                        currTree = TreeNode(i)
                        currTree.left = l
                        currTree.right = r
                        allTrees.append(currTree)
            
            return allTrees
        
        return generateTrees(1, n) if n else []
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



# [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

难度 困难

二叉搜索树中的两个节点被错误地交换。

请在不改变其结构的情况下，恢复这棵树。

**示例 1:**

```
输入: [1,3,null,null,2]

   1
  /
 3
  \
   2

输出: [3,1,null,null,2]

   3
  /
 1
  \
   2
```

**示例 2:**

```
输入: [3,1,4,null,null,2]

  3
 / \
1   4
   /
  2

输出: [2,1,4,null,null,3]

  2
 / \
1   4
   /
  3
```

**进阶:**

- 使用 O(*n*) 空间复杂度的解法很容易实现。
- 你能想出一个只使用常数空间的解决方案吗？



**解法**

Morris 中序遍历。

Morris 遍历算法整体步骤如下（假设当前遍历到的节点为 xx）：

1. 如果 $x$ 无左孩子，则访问 $x$ 的右孩子，即 $x = x.\textit{right}$ 。
2. 如果 $x$ 有左孩子，则找到 $x$ 左子树上最右的节点（即左子树中序遍历的最后一个节点，$x$ 在中序遍历中的前驱节点），我们记为 $\textit{predecessor}$。根据 $\textit{predecessor}$ 的右孩子是否为空，进行如下操作。
    + 如果 $\textit{predecessor}$ 的右孩子为空，则将其右孩子指向 $x$，然后访问 $x$ 的左孩子，即 $x = x.\textit{left}$ 。
    + 如果 $\textit{predecessor}$ 的右孩子不为空，则此时其右孩子指向 $x$，说明我们已经遍历完 $x$ 的左子树，我们将 $\textit{predecessor}$ 的右孩子置空，然后访问 $x$ 的右孩子，即 $x = x.\textit{right}$ 。
3. 重复上述操作，直至访问完整棵树。

时间复杂度： $O(N)$ ，其中 $N$ 为二叉搜索树的高度。空间复杂度： $O(1)$ 。



**代码**

``` python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

        x = None
        y = None
        pred = None 
        predecessor = None

        while root is not None:

            if root.left is not None:
                predecessor = root.left
                while predecessor.right is not None and predecessor.right != root:
                    predecessor = predecessor.right
                
                if predecessor.right is None:
                    predecessor.right = root 
                    root = root.left
                
                # 说明左子树已经访问完了，我们需要断开链接
                else:
                    if pred is not None and root.val < pred.val:
                        y = root 
                        if x is None:
                            x = pred
                    
                    pred = root 
                    predecessor.right = None
                    root = root.right
            
            # 如果没有左孩子，则直接访问右孩子
            else:
                if pred is not None and root.val < pred.val:
                    y = root 
                    if x is None:
                        x = pred 
                
                pred = root 
                root = root.right
        
        x.val, y.val = y.val, x.val
```





# [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

难度 简单

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

**示例 1:**

```
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
```

**示例 2:**

```
输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false
```

**示例 3:**

```
输入:       1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

输出: false
```



**解法**

递归判断左右子树是否相等。时间复杂度： $O(n)$ ，空间复杂度： $O(n)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:

        if p is None and q is None:
            return True
        
        if p is None or q is None or p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

