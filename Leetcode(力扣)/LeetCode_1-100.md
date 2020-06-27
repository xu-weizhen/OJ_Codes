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



# [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

难度 中等

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

 

**示例:**

```
给定 1->2->3->4, 你应该返回 2->1->4->3.
```



**解法**

依次交换两节点。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or head.next == None:
            return head

        def swapTwo(head, first, last):
            first.next = last.next
            last.next = first
            head.next = last
            return first
        
        headNode = ListNode(0)
        headNode.next = head
        h = headNode
        f = headNode.next
        l = f.next

        while True:
            h = swapTwo(h, f, l)
            f = h.next
            if f == None or f.next == None:
                break
            l = f.next
        
        return headNode.next
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



# [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

难度 简单

给定一个排序数组，你需要在**[ 原地](http://baike.baidu.com/item/原地算法)** 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

 

**示例 1:**

```
给定数组 nums = [1,1,2], 

函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

你不需要考虑数组中超出新长度后面的元素。
```

 

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```



**解法**

+ 方法一：直接删除重复元素。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。
+ 方法二：使用双指针移动元素位置。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        i = 0
        while i + 1 < len(nums):
            if nums[i] == nums[i + 1]:
                del nums[i + 1]
            else:
                i += 1
        
        return len(nums)

# 方法二
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        i = 0
        for j in range(i + 1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
        
        nums = nums[:i + 1]
        return len(nums)
```



# [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

难度 简单

给你一个数组 *nums* 和一个值 *val*，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 *val* 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 **[原地 ](https://baike.baidu.com/item/原地算法)修改输入数组**。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

 

**示例 1:**

```
给定 nums = [3,2,2,3], val = 3,

函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,1,2,2,3,0,4,2], val = 2,

函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

注意这五个元素可为任意顺序。

你不需要考虑数组中超出新长度后面的元素。
```

 

**解法**

使用双指针移动元素位置。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for j in range(len(nums)):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i
```



# [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

难度 简单

实现 [strStr()](https://baike.baidu.com/item/strstr/811469) 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回 **-1**。

**示例 1:**

```
输入: haystack = "hello", needle = "ll"
输出: 2
```

**示例 2:**

```
输入: haystack = "aaaaa", needle = "bba"
输出: -1
```

**说明:**

当 `needle` 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 `needle` 是空字符串时我们应当返回 0 。这与C语言的 [strstr()](https://baike.baidu.com/item/strstr/811469) 以及 Java的 [indexOf()](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html#indexOf(java.lang.String)) 定义相符。



**解法**：

+ 方法一： KMP算法。构建 `next` 数组，该数组表示在某一位时，当前后缀与前缀若相同，则当前位对应的是前缀的哪一位。如字符串 `abcabcd` 的 `next` 数组为 `[-1, 0, 0, 0, 1, 2, 0]` 。通过 `next` 数组，查找时被搜索字符串上的指针可以实现不回溯。时间复杂度： $O(M + N)$ ，空间复杂度： $O(N)$ 。$M,N$ 分别为字符串和模式串长度。
+ 方法二：Rabin-Karp算法。计算哈希值。计算字符串中长度为 $N$ 的子串的哈希值，与模式串哈希值做比较。时间复杂度： $O(M)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
# 方法一
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if needle == '':
                return 0

        nextList = self.buildNext(needle)
        s = 0
        p = 0

        while s < len(haystack) and p < len(needle):
            if p == -1 or haystack[s] == needle[p]:
                s += 1
                p += 1
            else:
                p = nextList[p]
        
        return s - len(needle) if p == len(needle) else -1


    def buildNext(self, needle):
        ans = [-1] * len(needle)
        head = -1
        tail = 0

        while tail < len(needle) - 1:
            if head == -1 or needle[head] == needle[tail]:
                head += 1
                tail += 1
                ans[tail] = head 
            else:
                head = ans[head]

        return ans

# 方法eer
# 官方题解
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        L, n = len(needle), len(haystack)
        if L > n:
            return -1
        
        # base value for the rolling hash function
        a = 26
        # modulus value for the rolling hash function to avoid overflow
        modulus = 2**31
        
        # lambda-function to convert character to integer
        h_to_int = lambda i : ord(haystack[i]) - ord('a')
        needle_to_int = lambda i : ord(needle[i]) - ord('a')
        
        # compute the hash of strings haystack[:L], needle[:L]
        h = ref_h = 0
        for i in range(L):
            h = (h * a + h_to_int(i)) % modulus
            ref_h = (ref_h * a + needle_to_int(i)) % modulus
        if h == ref_h:
            return 0
              
        # const value to be used often : a**L % modulus
        aL = pow(a, L, modulus) 
        for start in range(1, n - L + 1):
            # compute rolling hash in O(1) time
            h = (h * a - h_to_int(start - 1) * aL + h_to_int(start + L - 1)) % modulus
            if h == ref_h:
                return start

        return -1
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



# [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

难度 困难

给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。

 

**示例 1:**

```
输入: [1,2,0]
输出: 3
```

**示例 2:**

```
输入: [3,4,-1,1]
输出: 2
```

**示例 3:**

```
输入: [7,8,9,11,12]
输出: 1
```

 

**提示：**

你的算法的时间复杂度应为O(*n*)，并且只能使用常数级别的额外空间。



**解法**

将大于 $1$ 小于 $n$ 的数都移动到下标为其数值减1的位置，遍历数组，如某个位置的值不是下标加1，则找到答案。时间复杂度： $O(N)$ ，空间复杂度： $O(1)$ 。



**代码**

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        for i in range(0, n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        
        ans = n + 1
        for i in range(0, n):
            if nums[i] != i + 1:
                ans = i + 1
                break
        
        return ans
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

