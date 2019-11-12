/*
**链接：https://www.nowcoder.com/questionTerminal/1277c681251b4372bdef344468e4f26e
**描述：将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
**时间：2019.11.4
**思路：略
*/

class Solution {
public:
    int StrToInt(string str) {
        int len = str.size();
        if(len == 0)
            return 0;
        
        bool nagetive = false;
        
        int p = 0;
        if(str[p] == '-')
        {
            nagetive = true;
            p++;
        }
        else if (str[p] == '+')
            p++;
        
        long long result = 0;
        for(; p < len; p++)
        {
            if(str[p] < '0' || str[p] > '9')
                return 0;
            
            if(nagetive)
                result = result * 10 - (str[p] - '0');
            else
                result = result * 10 + (str[p] - '0');
            
            if(result > 2147483647 || result < -2147483648)
                return 0; 
        }
        
        return result;
    }
};