/*
**链接：https://www.nowcoder.com/questionTerminal/12d959b108cb42b1ab72cef4d36af5ec
**描述：汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
**时间：2019.10.31
**思路：略
*/

class Solution {
public:
    string LeftRotateString(string str, int n) {
        if(n > str.size())
            return str;
        
        string str1 = str.substr(0, n);
        string str2 = str.substr(n);
        return str2 + str1;
    }
};