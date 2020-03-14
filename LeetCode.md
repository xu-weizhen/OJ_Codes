[toc]



# [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

难度 简单

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```




**解法**


+ 解法一：使用两个循环判断和是否为`target`。

+ 解法二：循环一次列表， 判断`target-num[i]`对应的值是否在`num[i+1:]`。

+ 解法三：解法二思想上， 使用哈希表来存储`nums[:i]` ，优化程序。




**代码**

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



# [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

难度 中等

给出两个 **非空** 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 **逆序** 的方式存储的，并且它们的每个节点只能存储 **一位** 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```



**解法**

注意：最高位相加产生进位的情况



**代码**

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



# [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

难度 中等

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```



**解法**

+ 从头到尾扫描字符，将当前确定的最长子串的所有字符放进一个列表，... 时间复杂度：$O(n^2)$，空间复杂度：$O(m)$ m为字符集大小

+ 从头到尾扫描字符，将当前确定的最长子串的所有字符放进一个字典，遇到新字符时判断该字符是否在字典中。若在字典中，将上一次遇到该字符之前遇到的字符移出字典，判断当前子串是否为目前为止最长的子串，修改子串长度，继续向后扫描。时间复杂度：$O(\min(m,n))$，空间复杂度：$O(m)$，m为字符集大小



**代码**

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

> + push_back()：向容器中加入一个右值元素(临时对象)时，首先会调用构造函数构造这个临时对象，然后调用拷贝构造函数将这个临时对象放入容器中。原来的临时变量释放。
> + emplace_back()：  c++11加入。 在容器尾部添加一个元素，这个元素原地构造，不需要触发拷贝构造和转移构造。 

# [4. 寻找两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

难度 困难

给定两个大小为 m 和 n 的有序数组 `nums1` 和 `nums2`。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。

你可以假设 `nums1` 和 `nums2` 不会同时为空。

**示例 1:**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```

**示例 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```



**解法**

将两个数组的前半部分和后半部分分别存入两个数组 left 和 right。若 left 中最大值大于 right 中最小值，则交换这两个值，直到 left 中的值全部小于 right 中的值，从而得到中位数。时间复杂度为：$O(max(m,n)\log(\frac{m+n}{2}))$，空间复杂度为：$O(m+n)$。$m, n$为两个数组的长度。



**代码**

``` python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len_1 = len(nums1)
        len_2 = len(nums2)
        
        if len_1 == 0 or len_2 == 0:
            if len_2 == 0:
                nums2 = nums1
                len_2 = len_1
            if len_2 % 2 == 0:
                return (nums2[len_2 // 2] + nums2[len_2 // 2 - 1]) / 2
            else:
                return nums2[len_2 // 2]
            
        if len_1 % 2 == 0 and len_2 % 2 == 0:
            half_1 = len_1 // 2
            half_2 = len_2 // 2
        elif len_1 % 2 == 0 and len_2 % 2 != 0:
            half_1 = len_1 // 2
            half_2 = len_2 // 2 + 1
        else:
            half_1 = len_1 // 2 + 1
            half_2 = len_2 // 2

        left = nums1[:half_1] + nums2[:half_2]
        right = nums1[half_1:] + nums2[half_2:]

        left.sort()
        right.sort()

        while left[-1] > right[0]:
            left[-1], right[0] = right[0], left[-1]
            left.sort()
            right.sort()

        if (len_1 + len_2) % 2 == 0:
            return  (left[-1] + right[0]) / 2
        else:
            return left[-1]
```



# [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

难度 中等

给定一个字符串 `s`，找到 `s` 中最长的回文子串。你可以假设 `s` 的最大长度为 1000。

**示例 1：**

```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
```

**示例 2：**

```
输入: "cbbd"
输出: "bb"
```



**解法**

考虑字符串中的每个字符作为中心点，向两边扩展，寻找最长回文子串。时间复杂度：$O(n^2)$ ，空间复杂度：$O(1)$ 。



**代码**

``` python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        result = [0, 0]             # [长度，起始下标]
        for i in range(0, len(s)):
            res = self.extend(s, i)
            if res[0] > result[0]:
                result = res
        
        string = ""
        index = result[1]
        for i in range(0, result[0]):
            string += s[index]
            index += 1

        return string
    
    def extend(self, s, index):
        # 中心为一个字符
        length1 = 1
        index1 = index - 1
        index2 = index + 1
        while index1 >= 0 and index2 < len(s) and s[index1] == s[index2]:
            index1 -= 1
            index2 += 1
            length1 += 2
        res = [length1, index1 + 1]
        
        # 中心为两个字符
        length2 = 0
        if index + 1 < len(s) and s[index] == s[index + 1]:
            index1 = index - 1
            index2 = index + 2
            length2 = 2
            while index1 >= 0 and index2 < len(s) and s[index1] == s[index2]:
                index1 -= 1
                index2 += 1
                length2 += 2
        
        if length2 > res[0]:
            res = [length2, index1 + 1]
        return res
```



# [6. Z 字形变换](https://leetcode-cn.com/problems/zigzag-conversion/)

难度 中等

将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 `"LEETCODEISHIRING"` 行数为 3 时，排列如下：

```
L   C   I   R
E T O E S I I G
E   D   H   N
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"LCIRETOESIIGEDHN"`。

请你实现这个将字符串进行指定行数变换的函数：

```
string convert(string s, int numRows);
```

**示例 1:**

```
输入: s = "LEETCODEISHIRING", numRows = 3
输出: "LCIRETOESIIGEDHN"
```

**示例 2:**

```
输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:

L     D     R
E   O E   I I
E C   I H   N
T     S     G
```



**解法**

使用`numRows`个字符串，分别保存第`n`行的字符，将原字符串中的字符按题目要求依次分配到对应字符串，最后拼接起来。



**代码**

``` python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        result = []
        for i in range(numRows):
            result.append('')
        
        index = 0
        strde = 1
        for ch in s:
            result[index] += ch
            if index + strde < 0 or index + strde >= numRows:
                strde = -strde
            index += strde
        
        return ''.join(result)
```



# [7. 整数反转](https://leetcode-cn.com/problems/reverse-integer/)

难度 简单

给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

**示例 1:**

```
输入: 123
输出: 321
```

 **示例 2:**

```
输入: -123
输出: -321
```

**示例 3:**

```
输入: 120
输出: 21
```

**注意:**

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 $[−2^{31}, 2^{31} − 1]$。请根据这个假设，如果反转后整数溢出那么就返回 0。



**解法**

+ 方法一：使用范围更大的数据类型保存运算结果，判断结果是否在有效范围内。

  * 时间复杂度：$O(\log(x))$ ，$x$ 中大约有 $\log_{10}(x)$ 位数字。空间复杂度：$O(1)$。

+ 方法二：计算过程中判断该步计算后是否溢出。（官方解法）

  * 如果 $\text{rev} > \frac{INTMAX}{10}$ 那么 $temp = \text{rev} \cdot 10 + \text{pop}$ 一定会溢出。
  * 如果 $\text{rev} == \frac{INTMAX}{10}$ 那么只要 $\text{pop} > 7$ ， $temp = \text{rev} \cdot 10 + \text{pop}$ 就会溢出。

  * 时间复杂度：$O(\log(x))$ ，$x$ 中大约有 $\log_{10}(x)$ 位数字。空间复杂度：$O(1)$。

  

**代码**

``` cpp
// 方法一
#include "limits.h"
class Solution {
public:
    int reverse(int x) {

        bool neg = false;
        if(x < 0)
            neg = true;
            
        long long res = 0;
        while(x)
        {
            res = res * 10 + x % 10;
            x = x / 10;
        }

        if(res < INT_MIN || res > INT_MAX)
            return 0;
        else
            return int(res);
    }
};
```

``` python
# 方法一
class Solution:
    def reverse(self, x: int) -> int:
        neg = False
        if x < 0:
            neg = True
            x = -x
        
        res = 0
        while x:
            r = x % 10
            x = x // 10
            res = res * 10 + r
        
        if neg:
            res = -res
        
        if res >= -pow(2, 31) and res < (pow(2, 31) - 1):
            return res 
        else:
            return 0
```



# [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

难度 中等

请你来实现一个 `atoi` 函数，使其能将字符串转换成整数。

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

**说明：**

假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 $[−2^{31}, 2^{31} − 1]$。如果数值超过这个范围，请返回  INT_MAX $(2^{31} − 1)$ 或 INT_MIN $(−2^{31})$ 。

**示例 1:**

```
输入: "42"
输出: 42
```

**示例 2:**

```
输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
```

**示例 3:**

```
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
```

**示例 4:**

```
输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
```

**示例 5:**

```
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−2^31) 。
```



**解法**

+ 方法一：逐字符处理
+ 方法二：正则式匹配



**代码**

```python
# 方法一
class Solution:
    def myAtoi(self, str: str) -> int:
        s = str.strip()      # 去空格
        
        if len(s) <= 0:
            return 0

        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        symbol = ['-', '+']

        if s[0] not in number and s[0] not in symbol:
            return 0
        
        neg = False
        index = 0

        if s[0] in symbol:
            if s[0] == '-':
                neg = True
            index += 1
        
        result = 0
        while index < len(s):
            if s[index] in number:
                result = result * 10 + number.index(s[index])
            else:
                break
            index += 1

        if neg:
            result = -result

        INT_MAX = pow(2, 31) - 1
        INT_MIN = -pow(2, 31)

        if result >= INT_MIN and result <= INT_MAX:
            return result
        elif result < INT_MIN:
            return INT_MIN
        else:
            return INT_MAX

# 方法二
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)
```



> 调用函数时，函数参数中的 \* 用于解包，将列表转化为一个个的值，并以这些值作为函数的参数
>
> 例：
>
> ``` python
> def fun(num1, num2):
>     return num1 + num2
> 
> li = [1, 2]
> res = fun(*li)	# 3
> ```



# [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

难度 简单 

判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

**示例 1:**

```
输入: 121
输出: true
```

**示例 2:**

```
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```

**示例 3:**

```
输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。
```



**解法**

+ 方法一：转换为字符串进行判断。时间复杂度：$O(n)$  $n$ 为数值位数，空间复杂度：$O(1)$ 。
+ 方法二：翻转后半部分数值，与前半部分进行比较。时间复杂度：$O(\log_{10}{n})$ ，空间复杂度：$O(1)$ 。



**代码**

``` python
# 方法一
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        
        s = str(x)
        index1 = 0
        index2 = len(s) - 1

        result = True
        while index1 < index2:
            if s[index1] != s[index2]:
                result = False
                break
            index1 += 1
            index2 -= 1
        
        return result

    
# 方法二
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        
        r = 0
        while r < x:
            r = r * 10 + x % 10
            x = x // 10
        
        return r == x or r // 10 == x
```



# [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

难度 困难

给你一个字符串 `s` 和一个字符规律 `p`，请你来实现一个支持 `'.'` 和 `'*'` 的正则表达式匹配。

```
'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
```

所谓匹配，是要涵盖 **整个** 字符串 `s`的，而不是部分字符串。

**说明:**

- `s` 可能为空，且只包含从 `a-z` 的小写字母。
- `p` 可能为空，且只包含从 `a-z` 的小写字母，以及字符 `.` 和 `*`。

**示例 1:**

```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

**示例 2:**

```
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

**示例 3:**

```
输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

**示例 4:**

```
输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
```

**示例 5:**

```
输入:
s = "mississippi"
p = "mis*is*p*."
输出: false
```



**解法**

+ 方法一：使用函数库
+ 方法二：动态规划。时间复杂度：$O(mn)$  $m, n$ 为字符串和模式串长度，空间复杂度：$O(mn)$ 。



**代码**

```cpp
// 方法一
#include <regex>
class Solution {
public:
    bool isMatch(string s, string p) {
        regex pattern(p);
        return regex_match(s, pattern);
    }
};
```

``` python
# 方法一
import re
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if re.fullmatch(p, s):
            return True
        return False

# 方法二
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        status = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]

        status[-1][-1] = True

        for i in range(len(s), -1, -1):
            for j in range(len(p) - 1, -1, -1):
                match = i < len(s) and p[j] in {s[i], '.'}

                if j + 1 < len(p) and p[j + 1] == '*':
                    status[i][j] = status[i][j + 2] or match and status[i + 1][j]
                else:
                    status[i][j] = match and status[i + 1][j + 1]

        return status[0][0]
```



# [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

难度 中等

给你 *n* 个非负整数 *a*1，*a*2，...，*a*n，每个数代表坐标中的一个点 (*i*, *ai*) 。在坐标内画 *n* 条垂直线，垂直线 *i* 的两个端点分别为 (*i*, *ai*) 和 (*i*, 0)。找出其中的两条线，使得它们与 *x* 轴共同构成的容器可以容纳最多的水。

**说明：**你不能倾斜容器，且 *n* 的值至少为 2。

 

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

 

**示例：**

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
```



**解法**

使用两个指针指向容器两端，初始为 $a_0$ 和 $a_n$ ，移动指针，减小底部长度，同时计算面积。返回计算过程中面积的最大值。时间复杂度：$O(n)$ ，空间复杂度：$O(1)$ 。



**代码**

``` python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        r = len(height) - 1
        bottom = r - l
        area = bottom * min(height[l], height[r])

        while l < r:
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
            
            bottom -= 1
            area = max(area, bottom * min(height[l], height[r]))
        
        return area
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



# [面试题10.01. 合并排序的数组](https://leetcode-cn.com/problems/sorted-merge-lcci/)

难度 简单

给定两个排序后的数组 A 和 B，其中 A 的末端有足够的缓冲空间容纳 B。 编写一个方法，将 B 合并入 A 并排序。

初始化 A 和 B 的元素数量分别为 *m* 和 *n*。

**示例:**

```
输入:
A = [1,2,3,0,0,0], m = 3
B = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
```



**解法**

+ 将$B$复制到$A$尾部，再使用`sort()`函数。时间复杂度：$O((m+n)log(m+n))$，空间复杂度： $O(log(m+n)) $
+ 使用辅助空间$t$，将$A$、$B$按大小顺序放入$t$中，再复制回$A$。时间复杂度：$O(m+n)$，空间复杂度：$O(m+n)$
+ 从大到小，将$A$、$B$中较大的元素放入$A$尾部。时间复杂度：$O(m+n)$，空间复杂度：$O(1)$



**代码**

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



# [面试题57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

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



# [面试题59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

难度 简单

本题与 239 题相同



# [面试题59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

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

