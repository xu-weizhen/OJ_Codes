/*
**链接：https://www.nowcoder.com/questionTerminal/59ac416b4b944300b617d4f7f111b215
**描述：写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
**时间：2019.11.4
**思路：略
*/

class Solution {
public:
    int Add(int num1, int num2)
    {
        int sum, carry;
        do
        {
            sum = num1 ^ num2;
            carry = (num1 & num2) << 1;
            
            num1 = sum;
            num2 = carry;
        }
        while(carry != 0);
        
        return sum;
    }
};