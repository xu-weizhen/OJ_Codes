[toc]


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

# [1025. 除数博弈](https://leetcode-cn.com/problems/divisor-game/)

爱丽丝和鲍勃一起玩游戏，他们轮流行动。爱丽丝先手开局。

最初，黑板上有一个数字 `N` 。在每个玩家的回合，玩家需要执行以下操作：

选出任一 `x`，满足 `0 < x < N` 且 `N % x == 0` 。
用 `N - x` 替换黑板上的数字 `N` 。
如果玩家无法执行这些操作，就会输掉游戏。

只有在爱丽丝在游戏中取得胜利时才返回 `True`，否则返回 `false`。假设两个玩家都以最佳状态参与游戏。

 
**示例 1：**

```
输入：2
输出：true
解释：爱丽丝选择 1，鲍勃无法进行操作。
```

**示例 2：**

```
输入：3
输出：false
解释：爱丽丝选择 1，鲍勃也选择 1，然后爱丽丝无法进行操作。
```


提示：

1. `1 <= N <= 1000`


**解法**

找规律。当 `N = 4` 时，Alice 选择 `1`，按 `N = 3` 的结果，Bob 输，当 `N = 5`时，Alice 选择 `1`，按 `N = 4` 的结果，Alice 输。 …… 可得 `N` 为偶数时，Alice 赢，否则输。时间复杂度： $O(1))$ ，空间复杂度： $O(1)$ 


**代码**

```python
class Solution:
    def divisorGame(self, N: int) -> bool:
        return N % 2 == 0
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



# [1512. 好数对的数目](https://leetcode-cn.com/problems/number-of-good-pairs/)

难度 简单

给你一个整数数组 `nums` 。

如果一组数字 `(i,j)` 满足 `nums[i]` == `nums[j]` 且 `i` < `j` ，就可以认为这是一组 **好数对** 。

返回好数对的数目。

 

**示例 1：**

```
输入：nums = [1,2,3,1,1,3]
输出：4
解释：有 4 组好数对，分别是 (0,3), (0,4), (3,4), (2,5) ，下标从 0 开始
```

**示例 2：**

```
输入：nums = [1,1,1,1]
输出：6
解释：数组中的每组数字都是好数对
```

**示例 3：**

```
输入：nums = [1,2,3]
输出：0
```

 

**提示：**

- `1 <= nums.length <= 100`
- `1 <= nums[i] <= 100`



**解法**

统计每个数字出现的次数，根据等差公式计算出好对数的数目。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 



**代码**

```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        count = list(collections.Counter(nums).values())
        
        ans = 0
        for c in count:
            ans += (c * (c - 1) // 2)

        return ans
```



# [1513. 仅含 1 的子串数](https://leetcode-cn.com/problems/number-of-substrings-with-only-1s/)

难度中等2

给你一个二进制字符串 `s`（仅由 '0' 和 '1' 组成的字符串）。

返回所有字符都为 1 的子字符串的数目。

由于答案可能很大，请你将它对 10^9 + 7 取模后返回。

 

**示例 1：**

```
输入：s = "0110111"
输出：9
解释：共有 9 个子字符串仅由 '1' 组成
"1" -> 5 次
"11" -> 3 次
"111" -> 1 次
```

**示例 2：**

```
输入：s = "101"
输出：2
解释：子字符串 "1" 在 s 中共出现 2 次
```

**示例 3：**

```
输入：s = "111111"
输出：21
解释：每个子字符串都仅由 '1' 组成
```

**示例 4：**

```
输入：s = "000"
输出：0
```

 

**提示：**

- `s[i] == '0'` 或 `s[i] == '1'`
- `1 <= s.length <= 10^5`



**解法**

统计每次出现连续的 `1` 时， `1` 的个数，根据其个数可以确定其可得到多少个子串，再进行累加。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$



**代码**

```python
class Solution:
    def numSub(self, s: str) -> int:
        left = 0
        ans = 0
        
        while left < len(s):
            if s[left] == '1':
                right = left + 1
                while right < len(s) and s[right] == '1':
                    right += 1
                
                l = right - left
                
                ans += (l + 1)  * l // 2
                
                left = right + 1
                
            else:
                left += 1
        
        return ans % (1000000000 + 7)
```


# [1528. 重新排列字符串](https://leetcode-cn.com/problems/shuffle-string/)

难度 简单

给你一个字符串 `s` 和一个 长度相同 的整数数组 `indices` 。

请你重新排列字符串 `s` ，其中第 `i` 个字符需要移动到 `indices[i]` 指示的位置。

返回重新排列后的字符串。


**示例 1：**

![图片](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/07/26/q1.jpg)

```
输入：s = "codeleet", indices = [4,5,6,7,0,2,1,3]
输出："leetcode"
解释：如图所示，"codeleet" 重新排列后变为 "leetcode" 。
```

**示例 2：**

```
输入：s = "abc", indices = [0,1,2]
输出："abc"
解释：重新排列后，每个字符都还留在原来的位置上。
```

**示例 3：**

```
输入：s = "aiohn", indices = [3,1,4,2,0]
输出："nihao"
```

**示例 4：**

```
输入：s = "aaiougrt", indices = [4,0,2,6,7,3,1,5]
输出："arigatou"
```

**示例 5：**

```
输入：s = "art", indices = [1,0,2]
输出："rat"
```

**提示：**

+ `s.length == indices.length == n`
+ `1 <= n <= 100`
+ `s` 仅包含小写英文字母。
+ `0 <= indices[i] < n`
+ `indices` 的所有的值都是唯一的（也就是说，`indices` 是整数 `0` 到 `n - 1` 形成的一组排列）。


**解法**

创建一个新数组，根据 `indices` 将原字符串中字符填到新数组的对应位置。时间复杂度： $O(N)$ ， 空间复杂度： $O(N)$ 。

**代码**

```python
# python
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        ans = ['' for _ in range(len(s))]
        
        for ch, index in zip(s, indices):
            ans[index] = ch
        
        return ''.join(ans)
```

```cpp
// c++
class Solution {
public:
    string restoreString(string s, vector<int>& indices) {

        string ans(s.length(), ' ');
        // string ans = s;

        for(int i=0; i<indices.size(); i++)
            ans[indices[i]] = s[i];
        
        return ans;
    }
};
```



# [1529. 灯泡开关 IV](https://leetcode-cn.com/problems/bulb-switcher-iv/)

难度 中等

房间中有 `n` 个灯泡，编号从 `0` 到 `n-1` ，自左向右排成一行。最开始的时候，所有的灯泡都是 关 着的。

请你设法使得灯泡的开关状态和 `target` 描述的状态一致，其中 `target[i]` 等于 `1` 第 `i` 个灯泡是开着的，等于 `0` 意味着第 `i` 个灯是关着的。

有一个开关可以用于翻转灯泡的状态，翻转操作定义如下：

选择当前配置下的任意一个灯泡（下标为 `i` ）
翻转下标从 `i` 到 `n-1` 的每个灯泡
翻转时，如果灯泡的状态为 `0` 就变为 `1` ，为 `1` 就变为 `0` 。

返回达成 `target` 描述的状态所需的 **最少** 翻转次数。

 

**示例 1：**

```
输入：target = "10111"
输出：3
解释：初始配置 "00000".
从第 3 个灯泡（下标为 2）开始翻转 "00000" -> "00111"
从第 1 个灯泡（下标为 0）开始翻转 "00111" -> "11000"
从第 2 个灯泡（下标为 1）开始翻转 "11000" -> "10111"
至少需要翻转 3 次才能达成 target 描述的状态
```

**示例 2：**

```
输入：target = "101"
输出：3
解释："000" -> "111" -> "100" -> "101".
```

**示例 3：**

```
输入：target = "00000"
输出：0
```

**示例 4：**

```
输入：target = "001011101"
输出：5
```

**提示：**

+ `1 <= target.length <= 10^5`
+ `target[i] == '0'` 或者 `target[i] == '1'`


**解法**

从左到右依次调整，使灯泡满足条件。时间复杂度： $O(N)$ ， 空间复杂度： $O(1)$ 。

**代码**

```python
class Solution:
    def minFlips(self, target: str) -> int:
        light = False
        ans = 0
        
        for ch in target:
            if not light and ch == '0':
                continue
            elif light and ch == '1':
                continue
            else:
                ans += 1
                light = not light
        
        return ans
```