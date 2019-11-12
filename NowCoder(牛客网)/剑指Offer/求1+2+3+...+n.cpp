/*
**链接：https://www.nowcoder.com/questionTerminal/7a0da8fc483247ff8800059e12d7caf1
**描述：求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
**时间：2019.11.4
**思路：略
*/

class Solution {
public:
    int Sum_Solution(int n) {
        int sum = n;
        sum && (sum += Sum_Solution(n-1));
        return sum;
    }
};