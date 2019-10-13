/*
**链接：https://www.nowcoder.com/questionTerminal/72a5a919508a4251859fb2cfb987a0e6
**描述：我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
**时间：2019.8.8
**思路：
依旧是斐波那契数列
2*n的大矩形，和n个2*1的小矩形
其中target*2为大矩阵的大小
有以下几种情形：
1.target <= 0 大矩形为<= 2*0,return 0；
2.target = 1大矩形为2*1，只有一种摆放方法，return1；
3.target = 2 大矩形为2*2，有两种摆放方法，return2；
4.target = n 分为两步考虑：
	第一次摆放一块 2*1 的小矩阵，则摆放方法总共为f(target - 1)
	第一次摆放一块1*2的小矩阵，则摆放方法总共为f(target-2)
*/

class Solution {
public:
    int rectCover(int number) {
        if(number<=0)
            return 0;
        if(number==1)
            return 1;
        if(number==2)
            return 2;
        return rectCover(number-1)+rectCover(number-2);
    }
};