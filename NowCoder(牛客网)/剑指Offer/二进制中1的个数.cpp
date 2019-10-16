/*
**链接：https://www.nowcoder.com/questionTerminal/8ee967e43c2c4ec193b040ea7fbb10b8
**描述：输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
**时间：2019.10.14
**思路：略
*/

class Solution {
public:
     int  NumberOf1(int n) {
         unsigned mask = 1;
         int count = 0;
         while(mask)
         {
             if(n&mask)
                 count++;
             mask = mask<<1;
         }
         return count;
     }
};