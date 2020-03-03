# 1.两数之和

给定一个整数数组`nums`和一个目标值`target`，请你在该数组中找出和为目标值的那**两个**整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

示例:

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```



## 解法


+ 解法一：使用两个循环判断和是否为`target`。

+ 解法二：循环一次列表， 判断`target-num[i]`对应的值是否在`num[i+1:]`。

+ 解法三：解法二思想上， 使用哈希表来存储`nums[:i]` ，优化程序。




## 代码

``` python
class Solution:
    def twoSum(self, nums, target):
        hash_map = {}
        length = len(nums)
        for i in range(0, length):
            dif = target - nums[i]
            if dif in hash_map:
                return [hash_map[dif], i]
            else:
                hash_map[nums[i]] = i
        return []
```

``` c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> m;
        for(int i = 0; i < nums.size(); i++)
        {
            if(m.count(target - nums[i]))
            {
                return {m[target - nums[i]], i};
            }
            m[nums[i]] = i;
        }
        return {};
    }
};
```



## 结果

| 执行用时 | 内存消耗 | 语言  |
| :---- | ------- | ------- |
| 56 ms | 14.7 MB | Python3 |
| 8 ms | 12.3 MB | Cpp  |



# 2.两数相加

给出两个**非空**的链表用来表示两个非负的整数。其中，它们各自的位数是按照**逆序**的方式存储的，并且它们的每个节点只能存储**一位**数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例：

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```



## 解法

注意：最高位相加产生进位的情况



## 代码

``` C++
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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* p1 = l1;
        ListNode* p2 = l2;
        ListNode* p;
        ListNode* result = new ListNode(-1);
        ListNode* r = result;

        int carry = 0;
        while(p1 != NULL && p2 != NULL)
        {
            p = new ListNode((p1->val + p2->val + carry)%10);
            carry = (p1->val + p2->val + carry) / 10;
            r->next = p;
            r = r->next;
            p1 = p1->next;
            p2 = p2->next;
        }

        while(p1 != NULL)
        {
            p = new ListNode((p1->val + carry)%10);
            carry = (p1->val + carry) / 10;
            r->next = p;
            r = r->next;
            p1 = p1->next;
        }

        while(p2 != NULL)
        {
            p = new ListNode((p2->val + carry)%10);
            carry = (p2->val + carry) / 10;
            r->next = p;
            r = r->next;
            p2 = p2->next;
        }

        if(carry > 0)
        {
            p = new ListNode(carry);
            r->next = p;
        }

        r = result;
        result = result->next;
        delete r;

        return result;
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
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        p1 = l1
        p2 = l2
        result = ListNode(-1)
        pr = result
        carry = 0
        while p1 is not None and p2 is not None:
            pr.next = ListNode((p1.val + p2.val + carry) % 10)
            carry = (p1.val + p2.val + carry) // 10
            pr = pr.next
            p1 = p1.next
            p2 = p2.next

        while p1 is not None:
            pr.next = ListNode((p1.val + carry) % 10)
            carry = (p1.val + carry) // 10
            pr = pr.next
            p1 = p1.next
        
        while p2 is not None:
            pr.next = ListNode((p2.val + carry) % 10)
            carry = (p2.val + carry) // 10
            pr = pr.next
            p2 = p2.next
        
        if carry != 0:
            pr.next = ListNode(carry)

        result = result.next
        return result
```



## 结果

| 执行用时 | 内存消耗 | 语言    |
| :------- | :------- | :------ |
| 28 ms    | 72.5 MB  | Cpp     |
| 84 ms    | 13.4 MB  | Python3 |



# 3.无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:
```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

示例 2:
```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

示例 3:
```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```



## 解法

+ 从头到尾扫描字符，将当前确定的最长子串的所有字符放进一个列表，... 时间复杂度：$O(n^2)$，空间复杂度：$O(m)$ m为字符集大小

+ 从头到尾扫描字符，将当前确定的最长子串的所有字符放进一个字典，遇到新字符时判断该字符是否在字典中。若在字典中，将上一次遇到该字符之前遇到的字符移出字典，判断当前子串是否为目前为止最长的子串，修改子串长度，继续向后扫描。时间复杂度：$O(\min(m,n))$，空间复杂度：$O(m)$，m为字符集大小



## 代码

``` C++
// 方法一
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<char> v;
        vector<char>::iterator it;
        int result = 0;
        int index = 0;
        int head = 0;
        int length = 0;
        while(index < s.length())
        {
            it = find(v.begin(), v.end(), s[index]);
            if (it != v.end())
            {
                if(length > result)
                    result = length;
                
                while(s[head] != s[index])
                {
                    it = find(v.begin(), v.end(), s[head]);
                    v.erase(it);
                    head++;
                    length--;
                }
                head++;
            }
            else
            {
                v.push_back(s[index]);
                length ++;
            }
            index ++;
        }

        if(length > result)
            result = length;
        return result;
    }
};

// 方法二
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, char> m;
        int result = 0;
        int index = 0;
        int head = 0;
        int length = 0;
        while(index < s.length())
        {
            if (m.count(s[index]))
            {
                if(length > result)
                    result = length;
                
                while(s[head] != s[index])
                {
                    m.erase(s[head]);
                    head++;
                    length--;
                }
                head++;
            }
            else
            {
                m[s[index]] = s[index];
                length ++;
            }
            index ++;
        }

        if(length > result)
            result = length;
        return result;
    }
};
```

``` python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        result = 0
        now = 0
        head = 0        
        m = {}

        while now < len(s):
            if s[now] in m:
                result = max(result, now - head)

                while s[head] != s[now]:
                    del m[s[head]]
                    head += 1
                head += 1
            else:
                m[s[now]] = s[now]
            
            now += 1
        
        return max(result, now - head)
```



## 结果

|  方法  | 执行用时 | 内存消耗 |  语言   |
| :----: | :------: | :------: | :-----: |
|   /    |  124 ms  | 13.4 MB  | Python3 |
| 方法一 |  28 ms   |  9.7 MB  |   Cpp   |
| 方法二 |  40 ms   | 11.8 MB  |   Cpp   |

# [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

反转一个单链表。

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```



## 代码

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



## 结果

| 执行用时 | 内存消耗 |    语言    |
| :------: | :------: | :--------: |
|  80 ms   | 36.6 MB  | Javascript |
|   8 ms   | 10.2 MB  |    Cpp     |
|  36 ms   | 14.4 MB  |  Python3   |

#  [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

使用队列实现栈的下列操作：

- push(x) -- 元素 x 入栈
- pop() -- 移除栈顶元素
- top() -- 获取栈顶元素
- empty() -- 返回栈是否为空

**注意:**

- 你只能使用队列的基本操作-- 也就是 `push to back`, `peek/pop from front`, `size`, 和 `is empty` 这些操作是合法的。
- 你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
- 你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。



## 代码

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



## 结果

| 执行用时 | 内存消耗 | 语言    |
| :------- | :------- | :------ |
| 56 ms    | 13.4 MB  | Python3 |
| 4 ms     | 9.5 MB   | Cpp     |



# [面试题 10.01. 合并排序的数组](https://leetcode-cn.com/problems/sorted-merge-lcci/)

给定两个排序后的数组 A 和 B，其中 A 的末端有足够的缓冲空间容纳 B。 编写一个方法，将 B 合并入 A 并排序。

初始化 A 和 B 的元素数量分别为 *m* 和 *n*。

**示例:**

```
输入:
A = [1,2,3,0,0,0], m = 3
B = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```



## 解法

+ 将$B$复制到$A$尾部，再使用`sort()`函数。时间复杂度：$O((m+n)log(m+n))$，空间复杂度： $O(log(m+n)) $
+ 使用辅助空间$t$，将$A$、$B$按大小顺序放入$t$中，再复制回$A$。时间复杂度：$O(m+n)$，空间复杂度：$O(m+n)$
+ 从大到小，将$A$、$B$中较大的元素放入$A$尾部。时间复杂度：$O(m+n)$，空间复杂度：$O(1)$



## 代码

``` cpp
// 方法一
class Solution {
public:
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        for (int i = 1; i != n; ++i)
            A[m + i] = B[i];
        sort(A.begin(), A.end());
    }
};

// 方法三
class Solution {
public:
    void merge(vector<int>& A, int m, vector<int>& B, int n) {
        int index1 = m - 1;
        int index2  = n - 1;
        int tail = m + n - 1;
        int cur;
        while (index1 >= 0 || index2 >= 0) {
            if (index1 == -1)
                cur = B[index2--];
            else if (index2 == -1)
                cur = A[index1--];
            else if (A[index1] > B[index2])
                cur = A[index1--];
            else
                cur = B[index2--];
            A[tail--] = cur;
        }
    }
};
```

``` python
# 方法一
class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """
        Do not return anything, modify A in-place instead.
        """
        A[m:] = B
        A.sort()
```



## 结果

|  方法  | 执行用时 | 内存消耗 |  语言   |
| :----: | :------: | :------: | :-----: |
| 方法一 |  32 ms   | 13.4 MB  | Python3 |
| 方法一 |   4 ms   | 11.5 MB  |   Cpp   |
| 方法三 |   4 ms   | 11.5 MB  |   Cpp   |