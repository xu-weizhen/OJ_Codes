[toc]

# [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)

难度 中等

让我们一起来玩扫雷游戏！

给定一个代表游戏板的二维字符矩阵。 **'M'** 代表一个**未挖出的**地雷，**'E'** 代表一个**未挖出的**空方块，**'B'** 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的**已挖出的**空白方块，**数字**（'1' 到 '8'）表示有多少地雷与这块**已挖出的**方块相邻，**'X'** 则表示一个**已挖出的**地雷。

现在给出在所有**未挖出的**方块中（'M'或者'E'）的下一个点击位置（行和列索引），根据以下规则，返回相应位置被点击后对应的面板：

1. 如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 **'X'**。
2. 如果一个**没有相邻地雷**的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的**未挖出**方块都应该被递归地揭露。
3. 如果一个**至少与一个地雷相邻**的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量。
4. 如果在此次点击中，若无更多方块可被揭露，则返回面板。

 

**示例 1：**

```
输入: 

[['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]

Click : [3,0]

输出: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

解释:
```

**示例 2：**

```
输入: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Click : [1,2]

输出: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'X', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

解释:
```

 

**注意：**

1. 输入矩阵的宽和高的范围为 [1,50]。
2. 点击的位置只能是未被挖出的方块 ('M' 或者 'E')，这也意味着面板至少包含一个可点击的方块。
3. 输入面板不会是游戏结束的状态（即有地雷已被挖出）。
4. 简单起见，未提及的规则在这个问题中可被忽略。例如，当游戏结束时你不需要挖出所有地雷，考虑所有你可能赢得游戏或标记方块的情况。



**解法**

广度优先搜索。时间复杂度： $O(mn)$ ，空间复杂度： $O(mn)$ 。



**代码**

```python
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:

        self.board = board

        if self.board[click[0]][click[1]] == 'M':
            self.board[click[0]][click[1]] = 'X' 
        else:
            self.row = len(self.board)
            self.col = len(self.board[0])
            que = deque()
            que.append((click[0], click[1]))

            while que:
                x, y = que.popleft()
                if self.board[x][y] != "E":
                    continue
                bom = self.around(x, y)
                if bom == 0:
                    self.board[x][y] = 'B' 
                    for xx, yy in [[x - 1, y - 1], [x - 1,  y], [x - 1, y + 1], [x, y - 1], [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]:
                        if xx >= 0 and xx < self.row and yy >= 0 and yy < self.col and self.board[xx][yy] == 'E':
                            que.append((xx, yy))
                else:
                    self.board[x][y] = str(bom)

        return self.board
    
    def around(self, x, y):
        bom = 0
        for xx, yy in [[x - 1, y - 1], [x - 1,  y], [x - 1, y + 1], [x, y - 1], [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]:
            if xx >= 0 and xx < self.row and yy >= 0 and yy < self.col and self.board[xx][yy] == 'M':
                bom += 1
        return bom
```




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



# [546. 移除盒子](https://leetcode-cn.com/problems/remove-boxes/)

难度 困难

给出一些不同颜色的盒子，盒子的颜色由数字表示，即不同的数字表示不同的颜色。
你将经过若干轮操作去去掉盒子，直到所有的盒子都去掉为止。每一轮你可以移除具有相同颜色的连续 k 个盒子（k >= 1），这样一轮之后你将得到 `k*k` 个积分。
当你将所有盒子都去掉之后，求你能获得的最大积分和。

 

**示例：**

```
输入：boxes = [1,3,2,2,2,3,4,3,1]
输出：23
解释：
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
----> [1, 3, 3, 4, 3, 1] (3*3=9 分) 
----> [1, 3, 3, 3, 1] (1*1=1 分) 
----> [1, 1] (3*3=9 分) 
----> [] (2*2=4 分)
```

 

**提示：**

- `1 <= boxes.length <= 100`
- `1 <= boxes[i] <= 100`



**代码**

```cpp
// 官方题解
class Solution {
public:
    int dp[100][100][100];

    int removeBoxes(vector<int>& boxes) {
        memset(dp, 0, sizeof dp);
        return calculatePoints(boxes, 0, boxes.size() - 1, 0);
    }

    int calculatePoints(vector<int>& boxes, int l, int r, int k) {
        if (l > r) return 0;
        if (dp[l][r][k] != 0) return dp[l][r][k];
        while (r > l && boxes[r] == boxes[r - 1]) {
            r--;
            k++;
        }
        dp[l][r][k] = calculatePoints(boxes, l, r - 1, 0) + (k + 1) * (k + 1);
        for (int i = l; i < r; i++) {
            if (boxes[i] == boxes[r]) {
                dp[l][r][k] = max(dp[l][r][k], calculatePoints(boxes, l, i, k + 1) + calculatePoints(boxes, i + 1, r - 1, 0));
            }
        }
        return dp[l][r][k];
    }
};
```



# [557. 反转字符串中的单词 III](https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/)

难度 简单

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

 

**示例：**

```
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
```

 

**提示：**

- 在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。



**代码**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        l = s.split(' ')
        
        for i in range(len(l)):
            l[i] = l[i][::-1]
        
        return ' '.join(l)
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



# [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

难度 中等1

给定两个字符串 **s1** 和 **s2**，写一个函数来判断 **s2** 是否包含 **s1** 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的子串。

**示例1:**

```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```

 

**示例2:**

```
输入: s1= "ab" s2 = "eidboaoo"
输出: False
```

 

**注意：**

1. 输入的字符串只包含小写字母
2. 两个字符串的长度都在 [1, 10,000] 之间



**解法**

统计 `s1`  和 `s2` 子串中字符出现的次数，若相同，则找到。时间复杂度： $O(n)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        
        target = Counter(s1)
        
        length = len(s1)
        
        for i in range(0, len(s2) - length + 1):
            counter = Counter(s2[i : i + length])
            if counter == target:
                return True
        
        return False
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



# [632. 最小区间](https://leetcode-cn.com/problems/smallest-range-covering-elements-from-k-lists/)

难度 困难

你有 `k` 个升序排列的整数数组。找到一个**最小**区间，使得 `k` 个列表中的每个列表至少有一个数包含在其中。

我们定义如果 `b-a < d-c` 或者在 `b-a == d-c` 时 `a < c`，则区间 [a,b] 比 [c,d] 小。

**示例 1:**

```
输入:[[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
输出: [20,24]
解释: 
列表 1：[4, 10, 15, 24, 26]，24 在区间 [20,24] 中。
列表 2：[0, 9, 12, 20]，20 在区间 [20,24] 中。
列表 3：[5, 18, 22, 30]，22 在区间 [20,24] 中。
```

**注意:**

1. 给定的列表可能包含重复元素，所以在这里升序表示 >= 。
2. 1 <= `k` <= 3500
3. -105 <= `元素的值` <= 105
4. **对于使用Java的用户，请注意传入类型已修改为List<List<Integer>>。重置代码模板后可以看到这项改动。**



**解法**

使用最小堆，每次从堆中去除当前最小的数，并将该数对应的列表中的下一个数放入堆中，重复直到有一个列表完成操作。时间复杂度： $O(nk \log k)$ ，空间复杂度： $O(k)$ ，其中 $n$ 为所有列表平均长度， $k$ 为列表数量。



**代码**

```python
# 官方代码
class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        rangeLeft, rangeRight = -10**6, 10**6

        maxValue = max(vec[0] for vec in nums)

        priorityQueue = [(vec[0], i, 0) for i, vec in enumerate(nums)]

        heapq.heapify(priorityQueue)

        while True:
            minValue, row, idx = heapq.heappop(priorityQueue)

            if maxValue - minValue < rangeRight - rangeLeft:
                rangeLeft, rangeRight = minValue, maxValue

            if idx == len(nums[row]) - 1:
                break
                
            maxValue = max(maxValue, nums[row][idx + 1])
            heapq.heappush(priorityQueue, (nums[row][idx + 1], row, idx + 1))
        
        return [rangeLeft, rangeRight]
```



# [637. 二叉树的层平均值](https://leetcode-cn.com/problems/average-of-levels-in-binary-tree/)

难度 简单

给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。

 

**示例 1：**

```
输入：
    3
   / \
  9  20
    /  \
   15   7
输出：[3, 14.5, 11]
解释：
第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。
```

 

**提示：**

- 节点值的范围在32位有符号整数范围内。



**解法**

层次遍历，遍历到层的最后一个节点时计算该层的平均值。时间复杂度： $O(n)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:

        if not root:
            return []
            
        q = deque()
        q.append(root)
        
        end = len(q)
        ans = []
        summ = 0
        count = 0

        while q:
            node = q.popleft()
            end -= 1
            count += 1
            summ += node.val

            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            
            if end == 0:
                end = len(q)
                ans.append(summ / count)
                summ = 0
                count = 0
        
        return ans
```



# [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

难度 中等

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

 

**示例 1：**

```
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"
```

**示例 2：**

```
输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

 

**提示：**

- 输入的字符串长度不会超过 1000 。



**解法**

+ 动态规划。时间复杂度：$O(n^2)$，空间复杂度：$O(n^2)$ 。
+ 中心扩展。枚举每一个可能的回文中心，然后用两个指针分别向左右两边拓展，当两个指针指向的元素相同的时候就拓展，否则停止拓展。时间复杂度：$O(n^2)$，空间复杂度：$O(1)$ 。
+ Manacher 算法。时间复杂度：$O(n)$，空间复杂度：$O(n)$ 。





**代码**

```python
# 方法一
class Solution:
    def countSubstrings(self, s: str) -> int:

        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        ans = len(s)

        for i in range(len(s)):
            dp[0][i] = True

        for l in range(1, len(s)):
            for i in range(len(s)):
                end = i + l 	# 字符串结尾下标

                if end >= len(s):
                    continue
                
                if l == 1:
                    dp[l][i] = (s[i] == s[end])
                else:
                    dp[l][i] = dp[l - 2][i + 1] and (s[i] == s[end])
                
                if dp[l][i]:
                    ans += 1
                    
        return ans

# 方法二
class Solution:
    def countSubstrings(self, s: str) -> int:

        ans = 0

        for i in range(len(s)):

            left = right = i 
            while left >= 0 and right < len(s) and s[left] == s[right]:
                ans += 1
                left -= 1
                right += 1
            
            left = i
            right = i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                ans += 1
                left -= 1
                right += 1
                
        return ans
```

```cpp
# 方法三 官方代码
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size();
        string t = "$#";
        for (const char &c: s) {
            t += c;
            t += '#';
        }
        n = t.size();
        t += '!';

        auto f = vector <int> (n);
        int iMax = 0, rMax = 0, ans = 0;
        for (int i = 1; i < n; ++i) {
            // 初始化 f[i]
            f[i] = (i <= rMax) ? min(rMax - i + 1, f[2 * iMax - i]) : 1;
            // 中心拓展
            while (t[i + f[i]] == t[i - f[i]]) ++f[i];
            // 动态维护 iMax 和 rMax
            if (i + f[i] - 1 > rMax) {
                iMax = i;
                rMax = i + f[i] - 1;
            }
            // 统计答案, 当前贡献为 (f[i] - 1) / 2 上取整
            ans += (f[i] / 2);
        }

        return ans;
    }
};
```



# [657. 机器人能否返回原点](https://leetcode-cn.com/problems/robot-return-to-origin/)

难度 简单

在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 **(0, 0) 处结束**。

移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 `R`（右），`L`（左），`U`（上）和 `D`（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。

**注意：**机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。

 

**示例 1:**

```
输入: "UD"
输出: true
解释：机器人向上移动一次，然后向下移动一次。所有动作都具有相同的幅度，因此它最终回到它开始的原点。因此，我们返回 true。
```

**示例 2:**

```
输入: "LL"
输出: false
解释：机器人向左移动两次。它最终位于原点的左侧，距原点有两次 “移动” 的距离。我们返回 false，因为它在移动结束时没有返回原点。
```



**解法**

统计各个方向移动的次数，对比左移和右移，上移和下移次数是否相等。时间复杂度： $O(n)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        d = collections.defaultdict(int)
        for ch in moves:
            d[ch] += 1
        return d['U'] == d['D'] and d['L'] == d['R']
```



# [679. 24 点游戏](https://leetcode-cn.com/problems/24-game/)

难度困难

你有 4 张写有 1 到 9 数字的牌。你需要判断是否能通过 `*`，`/`，`+`，`-`，`(`，`)` 的运算得到 24。

**示例 1:**

```
输入: [4, 1, 8, 7]
输出: True
解释: (8-4) * (7-1) = 24
```

**示例 2:**

```
输入: [1, 2, 1, 2]
输出: False
```

**注意:**

1. 除法运算符 `/` 表示实数除法，而不是整数除法。例如 4 / (1 - 2/3) = 12 。
2. 每个运算符对两个数进行运算。特别是我们不能用 `-` 作为一元运算符。例如，`[1, 1, 1, 1]` 作为输入时，表达式 `-1 - 1 - 1 - 1` 是不允许的。
3. 你不能将数字连接在一起。例如，输入为 `[1, 2, 1, 2]` 时，不能写成 12 + 12 。



**解法**

递归穷举，枚举9216种计算过程，验证计算结果。时间复杂度： $O(1)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def judgePoint24(self, nums: List[int]) -> bool:
        epsilon = 1e-6

        def cal(numList):
            if len(numList) == 0:
                return False
            elif len(numList) == 1:
                return abs(numList[0] - 24.0) < epsilon
            
            ans = False
            for i, a in enumerate(numList):
                for j, b in enumerate(numList):
                    if i != j:
                        l = []

                        for k, c in enumerate(numList):
                            if k != i and k != j:
                                l.append(c)
                        
                        ans = cal(l + [a + b]) or cal(l + [a - b]) or cal(l + [a * b])
                        if not ans:
                            if abs(b) > epsilon:
                                ans = cal(l + [a / b])
                    
                        if ans:
                            return ans      
            return ans 
        
        return cal(nums)
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



# [696. 计数二进制子串](https://leetcode-cn.com/problems/count-binary-substrings/)

难度简单208

给定一个字符串 `s`，计算具有相同数量0和1的非空(连续)子字符串的数量，并且这些子字符串中的所有0和所有1都是组合在一起的。

重复出现的子串要计算它们出现的次数。

**示例 1 :**

```
输入: "00110011"
输出: 6
解释: 有6个子串具有相同数量的连续1和0：“0011”，“01”，“1100”，“10”，“0011” 和 “01”。

请注意，一些重复出现的子串要计算它们出现的次数。

另外，“00110011”不是有效的子串，因为所有的0（和1）没有组合在一起。
```

**示例 2 :**

```
输入: "10101"
输出: 4
解释: 有4个子串：“10”，“01”，“10”，“01”，它们具有相同数量的连续1和0。
```

**注意：**

- `s.length` 在1到50,000之间。
- `s` 只包含“0”或“1”字符。



**解法**

统计字符连续出现的次数，两个相邻次数中较小者，就是这一部分字符中符合要求的子串数量。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:

        if len(s) <= 1:
            return 0

        index  = 0
        ans = 0
        lastCount = None

        while index < len(s):
            ch = s[index]
            count = 0

            while index < len(s) and s[index] == ch:
                index += 1
                count += 1

            if lastCount:
                ans += min(lastCount, count)
            lastCount = count 
        
        return ans
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



# [733. 图像渲染](https://leetcode-cn.com/problems/flood-fill/)

难度 简单

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 `(sr, sc)` 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 `newColor`，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

**示例 1:**

```
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。
```

**注意:**

- `image` 和 `image[0]` 的长度在范围 `[1, 50]` 内。
- 给出的初始点将满足 `0 <= sr < image.length` 和 `0 <= sc < image[0].length`。
- `image[i][j]` 和 `newColor` 表示的颜色值在范围 `[0, 65535]`内。



**解法**

广度优先搜索或深度优先搜索。时间复杂度： $O(mn)$ ，空间复杂度： $O(mn)$ 。



**代码**

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        q = deque()
        q.append((sr, sc))

        oldColor = image[sr][sc]

        if oldColor == newColor:
            return image

        while q:
            xx, yy = q.popleft()
            image[xx][yy] = newColor

            for x, y in [(xx - 1, yy), (xx + 1, yy), (xx, yy - 1), (xx, yy + 1)]:
                if x >= 0 and x < len(image) and y >= 0 and y < len(image[0]) and image[x][y] == oldColor:
                    q.append((x, y))

        return image
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



# [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/)

难度 中等

给定一个无向图`graph`，当这个图为二分图时返回`true`。

如果我们能将一个图的节点集合分割成两个独立的子集A和B，并使图中的每一条边的两个节点一个来自A集合，一个来自B集合，我们就将这个图称为二分图。

`graph`将会以邻接表方式给出，`graph[i]`表示图中与节点`i`相连的所有节点。每个节点都是一个在`0`到`graph.length-1`之间的整数。这图中没有自环和平行边： `graph[i]` 中不存在`i`，并且`graph[i]`中没有重复的值。

```
示例 1:
输入: [[1,3], [0,2], [1,3], [0,2]]
输出: true
解释: 
无向图如下:
0----1
|    |
|    |
3----2
我们可以将节点分成两组: {0, 2} 和 {1, 3}。
示例 2:
输入: [[1,2,3], [0,2], [0,1,3], [0,2]]
输出: false
解释: 
无向图如下:
0----1
| \  |
|  \ |
3----2
我们不能将节点分割成两个独立的子集。
```

**注意:**

- `graph` 的长度范围为 `[1, 100]`。
- `graph[i]` 中的元素的范围为 `[0, graph.length - 1]`。
- `graph[i]` 不会包含 `i` 或者有重复的值。
- 图是无向的: 如果`j` 在 `graph[i]`里边, 那么 `i` 也会在 `graph[j]`里边。



**解法**

对节点进行涂色，若涂色过程中没有冲突，则是二分图。时间复杂度： $O(M+N)$ ，空间复杂度： $O(N)$ ， $M,N$ 分别为图中的边数和节点数。



**代码**

```python
# 官方题解
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        UNCOLORED, RED, GREEN = 0, 1, 2
        color = [UNCOLORED] * n
        valid = True

        def dfs(node: int, c: int):
            nonlocal valid
            color[node] = c
            cNei = (GREEN if c == RED else RED)
            for neighbor in graph[node]:
                if color[neighbor] == UNCOLORED:
                    dfs(neighbor, cNei)
                    if not valid:
                        return
                elif color[neighbor] != cNei:
                    valid = False
                    return

        for i in range(n):
            if color[i] == UNCOLORED:
                dfs(i, RED)
                if not valid:
                    break
        
        return valid
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



# [841. 钥匙和房间](https://leetcode-cn.com/problems/keys-and-rooms/)

难度 中等

有 `N` 个房间，开始时你位于 `0` 号房间。每个房间有不同的号码：`0，1，2，...，N-1`，并且房间里可能有一些钥匙能使你进入下一个房间。

在形式上，对于每个房间 `i` 都有一个钥匙列表 `rooms[i]`，每个钥匙 `rooms[i][j]` 由 `[0,1，...，N-1]` 中的一个整数表示，其中 `N = rooms.length`。 钥匙 `rooms[i][j] = v` 可以打开编号为 `v` 的房间。

最初，除 `0` 号房间外的其余所有房间都被锁住。

你可以自由地在房间之间来回走动。

如果能进入每个房间返回 `true`，否则返回 `false`。



**示例 1：**

```
输入: [[1],[2],[3],[]]
输出: true
解释:  
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。
```

**示例 2：**

```
输入：[[1,3],[3,0,1],[2],[0]]
输出：false
解释：我们不能进入 2 号房间。
```

**提示：**

1. `1 <= rooms.length <= 1000`
2. `0 <= rooms[i].length <= 1000`
3. 所有房间中的钥匙数量总计不超过 `3000`。



**解法**

深度优先搜索。时间复杂度： $O(m+n)$ ，空间复杂度： $O(n)$ ，其中 $m$ 为钥匙数量， $n$ 为房间数量。



**代码**

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        
        visited = [0] * len(rooms)

        def dfs(now):
            visited[now] = 1

            for key in rooms[now]:
                if visited[key] == 0:
                    dfs(key)
                    
        dfs(0)
        return sum(visited) == len(rooms)
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

 