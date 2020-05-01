# [面试题 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

难度 中等

地上有一个m行n列的方格，从坐标 `[0,0]` 到坐标 `[m-1,n-1]` 。一个机器人从坐标 `[0, 0] `的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

**示例 1：**

```
输入：m = 2, n = 3, k = 1
输出：3
```

**示例 1：**

```
输入：m = 3, n = 1, k = 0
输出：1
```

**提示：**

- `1 <= n,m <= 100`
- `0 <= k <= 20`



**解法**

广度优先搜索。时间复杂度： $O(MN)$ ，空间复杂度：  $O(MN)$ 。



**代码**

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        self.grid = [[0 for i in range(n)] for i in range(m)]
        self.dx = [-1, 1, 0, 0]
        self.dy = [0, 0, -1, 1]
        self.m = m
        self.n = n
        self.k = k

        self.grid[0][0] = 1
        self.move(0, 0)

        count = 0
        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j]:
                    count += 1
        
        return count

    def count(self, x, y):
        count = 0
        while x > 0:
            count += x % 10
            x //= 10
        while y > 0:
            count += y % 10
            y //= 10
        return count

    def move(self, x, y):
        for i in range(4):
            if 0 <= x + self.dx[i] < self.m and 0 <=  y + self.dy[i] < self.n \
                    and self.grid[x + self.dx[i]][y + self.dy[i]] == 0 \
                    and self.count(x + self.dx[i], y + self.dy[i]) <= self.k:
                self.grid[x + self.dx[i]][y + self.dy[i]] = 1
                self.move(x + self.dx[i], y + self.dy[i])
```



# [面试题 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

难度 简单

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

**示例 1：**

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

**示例 2：**

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

 

**限制：**

- `0 <= k <= arr.length <= 10000`
- `0 <= arr[i] <= 10000`



**解法**

+ 方法一：快速排序，取最小的 $k$ 位。时间复杂度：期望 $O(n)$ 最坏 $O(n^2)$，空间复杂度：期望 $O(\log n)$ 最坏 $O(n)$ 。
+ 方法二：构建最小堆，取 $k$ 次堆顶。时间复杂度：初始建堆 $O(n)$ 重建堆每次为 $O(\log n)$ ，空间复杂度：  $O(1)$ 。



**代码**

```python
# 方法二 使用heapq
import heapq
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        return heapq.nsmallest(k, arr)

# 方法二 手写堆排序
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        self.buildHeap(arr)
        res = []
        for i in range(k):
            res.append(self.pop(arr))
        return  res

    def buildHeap(self, arr):
        heap = arr
        for root in range(len(heap)-1 , -1, -1):
            child = root * 2 + 1
            while child < len(heap):
                if child + 1 < len(heap) and heap[child + 1] < heap[child]:
                    child += 1
                if heap[child] >= heap[root]:
                    break
                heap[child], heap[root] = heap[root], heap[child]
                root = child
                child = 2 * root + 1

    def pop(self, heap):
        res = heap[0]
        heap[0] = heap[-1]
        del heap[-1]

        root = 0
        child = root * 2 + 1
        while child < len(heap):
            if child + 1 < len(heap) and heap[child + 1] < heap[child]:
                child += 1
            if heap[child] > heap[root]:
                break

            heap[child], heap[root] = heap[root], heap[child]
            root = child
            child = 2 * root + 1

        return res
```

# [面试题 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

难度 困难

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

 

**示例 1:**

```
输入: [7,5,6,4]
输出: 5
```

 

**限制：**

```
0 <= 数组长度 <= 50000
```



**思路**

使用归并排序，在排序时计数逆序对数量。若归并时右边数字较小，则于该数字有关的逆序对数量为左边数组中还剩的数字的数量。时间复杂度： $O(N\log N)$ ，空间复杂度：$O(N)$ 。



**代码**

```python
# 官方题解
class Solution:
    def mergeSort(self, nums, tmp, l, r):
        if l >= r:
            return 0

        mid = (l + r) // 2
        inv_count = self.mergeSort(nums, tmp, l, mid) + self.mergeSort(nums, tmp, mid + 1, r)
        i, j, pos = l, mid + 1, l
        while i <= mid and j <= r:
            if nums[i] <= nums[j]:
                tmp[pos] = nums[i]
                i += 1
                inv_count += (j - (mid + 1))
            else:
                tmp[pos] = nums[j]
                j += 1
            pos += 1
        for k in range(i, mid + 1):
            tmp[pos] = nums[k]
            inv_count += (j - (mid + 1))
            pos += 1
        for k in range(j, r + 1):
            tmp[pos] = nums[k]
            pos += 1
        nums[l:r+1] = tmp[l:r+1]
        return inv_count

    def reversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        tmp = [0] * n
        return self.mergeSort(nums, tmp, 0, n - 1)
```



# [面试题 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

难度 中等

一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

 

**示例 1：**

```
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```

**示例 2：**

```
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
```

 

**限制：**

- `2 <= nums <= 10000`

 

**解法**

先对所有数字进行一次异或，得到两个出现一次的数字的异或值。在异或结果中找到任意为1的位。根据这一位对所有的数字进行分组。在每个组内进行异或操作，得到两个数字。时间复杂度：$O(N)$，空间复杂度：$O(1)$。



**代码**

```python
# 官方题解
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        ret = functools.reduce(lambda x, y: x ^ y, nums)
        div = 1
        while div & ret == 0:
            div <<= 1
        a, b = 0, 0
        for n in nums:
            if n & div:
                a ^= n
            else:
                b ^= n
        return [a, b]
```



# [面试题 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

难度 简单

输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

 

**示例 1：**

```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**示例 2：**

```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

 

**限制：**

- `1 <= target <= 10^5`



**解法**

双指针。时间复杂度：$O(target)$，空间复杂度：$O(1)$。



**代码**

``` python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        if target <= 2:
            return []

        index1 = 1
        index2 = 2
        s = index1 + index2
        result = []

        while index2 <= target / 2 + 1:
            if s == target:
                t = []
                for i in range(index1, index2 + 1):
                    t.append(i)
                result.append(t)
                index2 += 1
                s += index2
            
            if s < target:
                index2 += 1
                s += index2
            else:
                s -= index1
                index1 += 1
        
        return result
```



# [面试题 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

难度 简单

本题与 239 题相同



# [面试题 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

难度 中等

请定义一个队列并实现函数 `max_value` 得到队列里的最大值，要求函数`max_value`、`push_back` 和 `pop_front` 的**均摊**时间复杂度都是O(1)。

若队列为空，`pop_front` 和 `max_value` 需要返回 -1

**示例 1：**

```
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```

**示例 2：**

```
输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
```

 

**限制：**

- `1 <= push_back,pop_front,max_value的总操作数 <= 10000`
- `1 <= value <= 10^5`



**解法**

使用一个双端队列保存当前数组的递减最大值序列。



**代码**

``` python
import queue
class MaxQueue:

    def __init__(self):
        self.queue = queue.Queue()
        self.maxvalue = queue.deque()

    def max_value(self) -> int:
        return self.maxvalue[0] if self.maxvalue else -1

    def push_back(self, value: int) -> None:
        while self.maxvalue and self.maxvalue[-1] < value:
            self.maxvalue.pop()
        self.maxvalue.append(value)

        self.queue.put(value)

    def pop_front(self) -> int:
        if self.queue.empty():
            return -1
        
        ans = self.queue.get()
        if ans == self.maxvalue[0]:
            self.maxvalue.popleft()
        return ans
```



# [面试题 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

难度 简单

0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

 

**示例 1：**

```
输入: n = 5, m = 3
输出: 3
```

**示例 2：**

```
输入: n = 10, m = 17
输出: 2
```

 

**限制：**

- `1 <= n <= 10^5`
- `1 <= m <= 10^6`



**解法**

约瑟夫环问题

当人数为1时，安全位置为 $0$ 。

当人数为2时，安全位置为 $(0 + m) \mod 2$ 。

当人数为3时，安全位置为 $((0 + m) \mod 2 + m) \mod 3$ 。

...

 时间复杂度：$O(N)$，空间复杂度：$O(1)$。



**代码**

```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        ans = 0;
        for i in range(2, n + 1):
            ans = (ans + m) % i
        return ans
```


