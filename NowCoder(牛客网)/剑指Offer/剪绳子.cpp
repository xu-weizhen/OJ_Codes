/*
**链接：https://www.nowcoder.com/questionTerminal/57d85990ba5b440ab888fc72b0751bf8
**描述：给你一根长度为n的绳子，请把绳子剪成m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
**时间：2019.10.17
**思路：若剩下为4，切成2,2；若剩下大于3，切成3，x。
*/

class Solution {
public:
    int cutRope(int number) {
        if(number==2)
            return 1;
        if(number==3)
            return 2;
        
        int count = 0;
        int sum = 1;
        count = number/3;
        if(number%3==1)
        {
            count = number/3-1;
            for(int i=0; i<count; i++)
                sum *= 3;
            sum *= 4;
        }
        else if(number%3==2)
        {
            for(int i=0; i<count; i++)
                sum *= 3;
            sum *= 2;
        }
        else
        {
            for(int i=0; i<count; i++)
                sum *= 3;
        }
        return sum;
    }
};