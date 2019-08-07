/*
**链接：https://www.nowcoder.com/questionTerminal/8c82a5b80378478f9484d87d1c5f12a4
**描述：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
**时间：2019.8.7
**思路：略
*/

class Solution {
public:
    int jumpFloor(int number) {
        if(number == 0)
            return 0;
        if(number == 1)
            return 1;
        if(number == 2)
            return 2;
        return jumpFloor(number-1)+jumpFloor(number-2);
    }
};