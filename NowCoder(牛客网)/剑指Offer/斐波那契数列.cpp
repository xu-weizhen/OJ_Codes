/*
**链接：https://www.nowcoder.com/questionTerminal/c6c7742f5ba7442aada113136ddea0c3
**描述：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39
**时间：2019.8.8
**思路：略
*/

class Solution {
public:
    int Fibonacci(int n) {
        if(n==0)
            return 0;
        if(n==1)
            return 1;
        if(n==2)
            return 1;
        return Fibonacci(n-1)+Fibonacci(n-2);
    }
};