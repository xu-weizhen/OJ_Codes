/*
**链接：https://www.nowcoder.com/questionTerminal/22243d016f6b47f2a6928b4313c85387
**描述：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
**时间：2019.8.7
**思路：使用数组保存到当前台阶跳法数。从当前台阶出发，对于可以到达的台阶，可到达该台阶的跳法数为：到达该台阶已计算出跳法数+到达当前台阶跳法数
*/

class Solution {
public:
    int jumpFloorII(int number) {
        int *sum = (int*)malloc(sizeof(int)*number+1);
        for(int i=0; i<=number; i++)
            sum[i]=1;
        int now = 1;		//当前台阶
        while(now != number)
        {
            for(int j=1; now+j<=number; j++)
                sum[now+j] += sum[now];
            now++;
        }
        return sum[number];
    }
};