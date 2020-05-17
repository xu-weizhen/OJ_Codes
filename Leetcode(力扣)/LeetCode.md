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



# [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

难度 中等

罗马数字包含以下七种字符： `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做 `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

**示例 1:**

```
输入: 3
输出: "III"
```

**示例 2:**

```
输入: 4
输出: "IV"
```

**示例 3:**

```
输入: 9
输出: "IX"
```

**示例 4:**

```
输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
```

**示例 5:**

```
输入: 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```



**解法**

+ 方法一：按照转换关系逐位转换。时间复杂度：$O(n)$ ，空间复杂度：$O(1)$ ， $n$ 待转换数字位数。
+ 方法二：贪心算法。每次减去一个已知罗马数字表示方法的值。时间复杂度：$O(1)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def intToRoman(self, num: int) -> str:
        self.result =''

        def part(num, char1, char2, char3):
            if num <= 3:
                self.result += char1 * num
            elif num == 4:
                self.result += char1 + char2
            elif num < 9:
                self.result += char2 + char1 * (num - 5)
            elif num == 9:
                self.result += char1 + char3

        if num // 1000 > 0:
            self.result += 'M' * (num // 1000)
            num %= 1000

        if num // 100 > 0:
            part(num // 100, 'C', 'D', 'M')
            num %= 100
            
        if num // 10 > 0:
            part(num // 10, 'X', 'L', 'C')
            num %= 10

        part(num, 'I', 'V', 'X')

        return self.result
            

# 方法二
class Solution:
    def intToRoman(self, num: int) -> str:
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

        index = 0
        res = ''
        while index < len(nums):
            while num >= nums[index]:
                res += romans[index]
                num -= nums[index]
            index += 1
        return res
```



# [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

难度 简单

罗马数字包含以下七种字符: `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做 `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

**示例 1:**

```
输入: "III"
输出: 3
```

**示例 2:**

```
输入: "IV"
输出: 4
```

**示例 3:**

```
输入: "IX"
输出: 9
```

**示例 4:**

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```

**示例 5:**

```
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```



**解法**

将罗马数字及其对应阿拉伯数字保存在字典中，通过查字典进行转换。时间复杂度：$O(n)$  $n$ 为字符串长度，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        d = {'M': 1000, 'CM': 900, 'D':500, 'CD': 400, 'C':100, 'XC':90, 'L':50, 'XL':40, 'X':10, 'IX':9, 'V':5, 'IV':4, 'I':1}

        res = 0
        index = 0
        while index < len(s):
            t = d.get(s[index:min(index + 2, len(s))])
            if t:
                res += t 
                index += 2
            else:
                res += d.get(s[index])
                index += 1
        return res
```



# [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

难度 简单

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

**示例 1:**

```
输入: ["flower","flow","flight"]
输出: "fl"
```

**示例 2:**

```
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```

**说明:**

所有输入只包含小写字母 `a-z` 。



**解法**

将当前得到的最长前缀依次与各个字符串进行比较。时间复杂度：$O(S)$  $S$ 是所有字符串中字符数量的总和，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ''

        res = strs[0]
        index = 1
        while index < len(strs):
            i = 0
            while i < len(res) and i < len(strs[index]):
                if res[i] == strs[index][i]:
                    i += 1
                else:
                    res = res[:i]
                    break
            res = res[:i]
            index += 1
        return res
```



# [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

难度 中等

给你一个包含 *n* 个整数的数组 `nums`，判断 `nums` 中是否存在三个元素 *a，b，c ，*使得 *a + b + c =* 0 ？请你找出所有满足条件且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

 

**示例：**

```
给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```



**解法**

对数组进行排序，一次固定一位，使用两个指针指向该位的后一位和数组最后一位，移动两个指针，验证三数之和。如果指针移动前后数值相等，则要跳过该值。时间复杂度：$O(n^2)$ ，空间复杂度：$O(1)$ （不含排序） / $O(n)$  （python 中 `sort` 的空间复杂度）。



**代码**

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums) < 3:
            return []

        nums.sort()
        res = []

        for i in range(0, len(nums) - 2):
            L = i + 1
            R = len(nums) - 1

            if i > 0 and nums[i] == nums[i - 1]:
                continue

            while L < R:
                if nums[i] + nums[L] + nums[R] == 0:
                    res.append([nums[i], nums[L], nums[R]])
                    L += 1
                    R -= 1
                    while L < R and nums[L] == nums[L - 1]:
                        L += 1
                    while R > L and nums[R] == nums[R + 1]:
                        R -= 1
                elif nums[i] + nums[L] + nums[R] > 0:
                    R -= 1
                else:
                    L += 1
        
        return res
```



# [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

难度 中等

给定一个包括 *n* 个整数的数组 `nums` 和 一个目标值 `target`。找出 `nums` 中的三个整数，使得它们的和与 `target` 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```



**解法**

对数组进行排序，一次固定一位，使用两个指针指向该位的后一位和数组最后一位，移动两个指针，计算三数之和与目标的差值。时间复杂度：$O(n^2)$ ，空间复杂度：$O(1)$ （不含排序） / $O(n)$  （python 中 `sort` 的空间复杂度）。



**代码**

```python
import sys
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        ans = sys.maxsize
        for i in range(0, len(nums) - 2):
            L = i + 1
            R = len(nums) - 1
            while L < R:
                distance = nums[i] + nums[L] + nums[R] - target
                if abs(ans - target) > abs(distance):
                    ans = nums[i] + nums[L] + nums[R]

                if distance > 0:
                    R -= 1
                else:
                    L += 1

        return ans
```



# [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

难度 中等

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png)

**示例:**

```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

**说明:**
尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。



**解法**

递归。时间复杂度： $O(3^N \times 4^M)$ 其中 $N$ 是输入数字中对应 3 个字母的数目， $M$ 是输入数字中对应 4 个字母的数目， 空间复杂度：$O(3^N \times 4^M)$ 。



**代码**

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        self.result = []
        self.dic = {'2': 'abc', '3':'def', '4':'ghi', '5':'jkl', '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
        self.makeString('', digits)
        return self.result


    def makeString(self, string, digits):
        if digits == '':
            self.result.append(string)
            return 0
        
        for ch in self.dic[digits[0]]:
            self.makeString(string + ch, digits[1:])
```



# [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

难度 中等

给定一个包含 *n* 个整数的数组 `nums` 和一个目标值 `target`，判断 `nums` 中是否存在四个元素 *a，**b，c* 和 *d* ，使得 *a* + *b* + *c* + *d* 的值与 `target` 相等？找出所有满足条件且不重复的四元组。

**注意：**

答案中不可以包含重复的四元组。

**示例：**

```
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```



**解法**

对数组进行排序，在15题的基础上，每次固定两位，使用两个指针指向该位的后一位和数组最后一位，移动两个指针，验证三数之和。如果指针移动前后数值相等，则要跳过该值。时间复杂度：$O(n^3)$ ，空间复杂度：$O(1)$ （不含排序） / $O(n)$  （python 中 `sort` 的空间复杂度）。



**代码**

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()

        res = []
        i = 0
        while i < len(nums) - 3:
            j = i + 1

            while j < len(nums) - 2:
                L = j + 1
                R = len(nums) - 1

                while L < R:
                    if nums[i] + nums[j] + nums[L] + nums[R] == target:
                        res.append([nums[i], nums[j], nums[L], nums[R]])
                    
                    if nums[i] + nums[j] + nums[L] + nums[R] > target:
                        R -= 1
                        while L < R and nums[R + 1] == nums[R]:
                            R -= 1
                    else:
                        L += 1
                        while L < R and nums[L - 1] == nums[L]:
                            L += 1

                j += 1
                while j < len(nums) - 2 and nums[j] == nums[j - 1]:
                    j += 1
            
            i += 1
            while i < len(nums) - 3 and nums[i] == nums[i-1]:
                i += 1

        return res
```



# [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

难度 中等

给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**说明：**

给定的 *n* 保证是有效的。

**进阶：**

你能尝试使用一趟扫描实现吗？



**解法**

使用两个指针，第二个指针指向第一个指针后 $n+1$ 位，两个指针同时后移，当第二个指针为空时，第一个指针指向被删除节点的前一个节点，即可将目标节点删除。时间复杂度：$O(N)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        i1 = head
        i2 = head

        for i in range(0, n):
            i2 = i2.next
        
        # 删除的是头节点
        if not i2:
            return head.next
        
        # 删除的不是头节点
        i2 = i2.next
        while i2:
            i2 = i2.next
            i1 = i1.next
        
        i1.next = i1.next.next
        return head
```



# [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

难度 简单

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

注意空字符串可被认为是有效字符串。

**示例 1:**

```
输入: "()"
输出: true
```

**示例 2:**

```
输入: "()[]{}"
输出: true
```

**示例 3:**

```
输入: "(]"
输出: false
```

**示例 4:**

```
输入: "([)]"
输出: false
```

**示例 5:**

```
输入: "{[]}"
输出: true
```



**解法**

使用栈进行括号的匹配。时间复杂度： $O(N)$ ，空间复杂度： $O(N)$ 。



**代码**

```python
class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        for ch in s:
            if ch in ['(', '[', '{']:
                st.append(ch)
            elif not st:
                return False
            elif ch == ')':
                if st[-1] == '(':
                    del st[-1]
                else:
                    return False
            elif ch == ']':
                if st[-1] == '[':
                    del st[-1]
                else:
                    return False
            elif ch == '}':
                if st[-1] == '{':
                    del st[-1]
                else:
                    return False
        if not st:
            return True
        else:
            return False
```

```cpp
#include <stack>

class Solution {
public:
    bool isValid(string s) {
        stack<char> st;

        for(int i=0; i<s.length(); i++)
        {
            if(s[i] == '(' || s[i] == '[' || s[i] == '{')
                st.push(s[i]);
            else
            {
                if(st.empty())
                    return false;
                else if(s[i] == ')')
                {
                    if(st.top() == '(')
                        st.pop();
                    else
                        return false;
                }
                else if (s[i] == ']')
                {
                    if(st.top() == '[')
                        st.pop();
                    else
                        return false;
                }
                else if (s[i] == '}')
                {
                    if(st.top() == '{')
                        st.pop();
                    else
                        return false;
                }
            }
        }

        if(st.empty())
            return true;
        else
            return false;
    }
};
```



# [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

难度 简单

将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

**示例：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```



**解法**

顺序遍历两个列表，将值较小的节点添加到新链表中。时间复杂度：$O(M+N)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode()
        tail = head

        while l1 and l2:
            if l1.val < l2.val:
                tail.next, l1 = l1, l1.next 
            else:
                tail.next, l2 = l2, l2.next 
            tail = tail.next
        
        while l1:
            tail.next, l1 = l1, l1.next 
            tail = tail.next
        
        while l2:
            tail.next, l2 = l2, l2.next 
            tail = tail.next
        
        return head.next
```



# [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

难度 中等

给出 *n* 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且**有效的**括号组合。

例如，给出 *n* = 3，生成结果为：

```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```



**解法**

递归。判断当前字符串中左右括号数量，若还能添加左括号，则添加左括号后递归；若右括号少于左括号，则添加右括号后递归。 时间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$， 空间复杂度：$O(\dfrac{4^n}{\sqrt{n}})$ 。



**代码**

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []

        def add(s, left, right):
            if len(s) == 2 * n:
                ans.append(s)
                return
            if left < n:
                add(s + '(', left + 1, right)
            if right < left:
                add(s + ')', left, right + 1)
        
        add('', 0, 0)
        return ans
```



# [23. 合并K个排序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

难度 困难

合并 *k* 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。

**示例:**

```
输入:
[
  1->4->5,
  1->3->4,
  2->6
]
输出: 1->1->2->3->4->4->5->6
```



**解法**

相邻的两个列表两两合并。时间复杂度： $O(N \log k)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if len(lists) <= 0:
            return None
        elif len(lists) == 1:
            return lists[0]
        else:
            while len(lists) > 1:
                res = self.merge2(lists[0], lists[1])
                del lists[0]
                del lists[0]
                lists.append(res)
            return lists[0]


    def merge2(self, list1, list2):
        res = ListNode(0)
        tail = res

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        
        while list1:
            tail.next = list1
            list1 = list1.next
            tail = tail.next
        
        while list2:
            tail.next = list2
            list2 = list2.next
            tail = tail.next

        tail.next = None
        return res.next
```



# [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

难度 困难

给你一个链表，每 *k* 个节点一组进行翻转，请你返回翻转后的链表。

*k* 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 *k* 的整数倍，那么请将最后剩余的节点保持原有顺序。

 

**示例：**

给你这个链表：`1->2->3->4->5`

当 *k* = 2 时，应当返回: `2->1->4->3->5`

当 *k* = 3 时，应当返回: `3->2->1->4->5`

 

**说明：**

- 你的算法只能使用常数的额外空间。
- **你不能只是单纯的改变节点内部的值**，而是需要实际进行节点交换。



**解法**

模拟法。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 官方题解
class Solution:
    # 翻转一个子链表，并且返回新的头与尾
    def reverse(self, head: ListNode, tail: ListNode):
        prev = tail.next
        p = head
        while prev != tail:
            nex = p.next
            p.next = prev
            prev = p
            p = nex
        return tail, head

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        hair = ListNode(0)
        hair.next = head
        pre = hair

        while head:
            tail = pre
            # 查看剩余部分长度是否大于等于 k
            for i in range(k):
                tail = tail.next
                if not tail:
                    return hair.next
            nex = tail.next
            head, tail = self.reverse(head, tail)
            # 把子链表重新接回原链表
            pre.next = head
            tail.next = nex
            pre = tail
            head = tail.next
        
        return hair.next
```




# [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

难度 中等

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 `[0,1,2,4,5,6,7]` 可能变为 `[4,5,6,7,0,1,2]` )。

搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 `-1` 。

你可以假设数组中不存在重复的元素。

你的算法时间复杂度必须是 *O*(log *n*) 级别。

**示例 1:**

```
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
```

**示例 2:**

```
输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
```



**思路**

二分查找。对有序的部分二分查找判断目标值是否在这一部分里，如果不在则在另一部分查找。时间复杂度： $O(\log N)$ ，空间复杂度：$O(1)$ 。



**代码**

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        if not nums:
            return -1

        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]:
                return mid
            
            # 左边有序
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            # 右边有序
            else:
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1

        return -1
```



# [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

难度 困难

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png)

上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 **感谢 Marcos** 贡献此图。

**示例:**

```
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```



**解法**

+ 方法一：设当前水面高度为 $i$ ，对于高度少于 $i$ 的地方，如果其左右均有高于等于 $i$ 的壁，则可以积水。（超时）时间复杂度： $O(iN^2)$  $i$ 为壁最高高度，空间复杂度： $O(1)$ 。
+ 方法二：暴力法。对每个横坐标，查找其左右最高壁的高度，其中较低者与当前高度只差为该处可积的水量。时间复杂度： $O(N^2)$ ，空间复杂度： $O(1)$ 。
+ 方法上：双指针。若左边壁小于右边壁，对于左边指针指向的壁，若小于目前左边壁的最大值，则该处可积的水量为该处高度与左边壁最大值之差；若大于目前左边壁最大值，则更新左边壁最大值。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def trap(self, height: List[int]) -> int:
        ans  = 0

        if not height:
            return ans
            
        for i in range(1, max(height) + 1):
            j = 0
            while j < len(height):
                if height[j] < i:
                    left = j - 1
                    while left >= 0 and height[left] < i:
                        left -= 1
                    # 存在左壁
                    if left >= 0:
                        right = j + 1
                        while right < len(height) and height[right] < i:
                            right += 1
                        
                        # 存在右壁
                        if right < len(height):
                            # print('height:{} left:{} right:{}'.format(i, left, right))
                            ans += (right - left - 1)
                            j = right + 1
                            continue
                j += 1
        return ans

# 方法二
class Solution:
    def trap(self, height: List[int]) -> int:
        ans  = 0

        if not height:
            return ans

        for i in range(1, len(height) - 1):
            left = max(height[:i])
            right = max(height[i + 1:])
            if left > height[i] and right > height[i]:
                ans += min(left, right) - height[i]
            
        return ans

# 方法三
class Solution:
    def trap(self, height: List[int]) -> int:
        ans  = 0
        left = 0
        right = len(height) - 1
        left_max = 0
        right_max = 0

        while left < right:
            if height[left] < height[right]:
                if height[left] > left_max:
                    left_max = height[left]
                else:
                    ans += left_max - height[left]
                
                left += 1
            else:
                if height[right] > right_max:
                    right_max = height[right]
                else:
                    ans  += right_max - height[right]
                
                right -= 1

        return ans
```



# [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

难度 困难

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

**示例:**

```
输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

**说明:**

假设你总是可以到达数组的最后一个位置。



**解法**

遍历数组中每个数，保存当前轮次的结尾位置和能跳到的最远位置。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)

        if n == 1:
            return 0

        farest = nums[0]
        end = 0
        ans = 1

        for i in range(n - 1):
            farest = max(farest, i + nums[i])
            if farest >= n - 1:
                break
            
            if i == end:
                ans += 1
                end = farest
            
        return ans
```



# [46. 全排列](https://leetcode-cn.com/problems/permutations/)

难度 中等

给定一个 **没有重复** 数字的序列，返回其所有可能的全排列。

**示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```



**思路**

将原数组划分为两部分，左边为已经填入答案数组的，右边为还没填入答案数组的，递归将右边的数分别填入。时间复杂度：$O(N*N!)$ ，空间复杂度：$O(N)$ 。



**代码**

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []

        def trackback(now):
            # 已全部填入
            if now == n:
                res.append(nums[:])
            
            for i in range(now, n):
                # 维护数组
                nums[now], nums[i] = nums[i], nums[now]
                # 填下一位
                trackback(now + 1)
                # 撤销操作
                nums[now], nums[i] = nums[i], nums[now]

        trackback(0)
        return res
```



# [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

难度 中等

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数。

**示例 1:**

```
输入: 2.00000, 10
输出: 1024.00000
```

**示例 2:**

```
输入: 2.10000, 3
输出: 9.26100
```

**示例 3:**

```
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```

**说明:**

- -100.0 < *x* < 100.0
- *n* 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。



**解法**

+ 方法一：直接计算
+ 方法二：快速幂。$x→x^2→x^{24}→x^{8}→x^{16}→x^{32}→x^{64}$ ， $x→x^2→x^{4}→x^{9}→x^{19}→x^{38}→x^{77}$ 。时间复杂度： $O(\log N)$ ，空间复杂度： $O(\log N)$。



**代码**

```python
# 方法一
class Solution:
    def myPow(self, x: float, n: int) -> float:
        return x ** n
    
# 方法二
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def quickMul(N):
            if N == 0:
                return 1.0
            y = quickMul(N // 2)
            return y * y if N % 2 == 0 else y * y * x
        
        return quickMul(n) if n >= 0 else 1.0 / quickMul(-n)
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


