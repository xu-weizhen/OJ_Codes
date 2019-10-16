/*
**链接：https://www.nowcoder.com/questionTerminal/1a834e5e3e1a4b7ba251417554e07c00
**描述：给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
保证base和exponent不同时为0
**时间：2019.10.14
**思路：略
*/

class Solution {
public:
    double Power(double base, int exponent) {
        if(Equal(base,0.0))
            return 0;
        
        if(exponent==0)
            return 1;
        
        bool nagtive = false;
        if(exponent<0)
        {
            nagtive = true;
            exponent = -exponent;
        }
        
        double result = 1.0;
        for(int i=0; i<exponent; i++)
            result *= base;
        
        if(nagtive)
            result = 1.0/result;
        
        return result;
    }
    
private:
    bool Equal(double a, double b)
    {
        if(a-b>-0.0000001 && a-b<0.0000001)
            return true;
        else
            return false;
    }
};