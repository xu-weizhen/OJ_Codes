/*
**链接：https://www.nowcoder.com/questionTerminal/6f8c901d091949a5837e24bb82a731f2
**描述：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
**时间：2019.10.28
**思路：略
*/

class Solution {
public:
    bool isNumeric(char* string)
    {
        if(string == nullptr)
            return false;
        
        bool numeric = scanInteger(&string);
        
        if(*string == '.')
        {
            ++string;
            numeric = scanUnsignedInteger(&string) || numeric;
        }
        
        if(*string == 'e' || *string == 'E')
        {
            ++string;
            numeric = scanInteger(&string) && numeric;
        }
        
        return numeric && (*string == '\0');
    }
    
    bool scanUnsignedInteger(char** str)
    {
        const char* before = *str;
        while(**str != '\0' && **str >= '0' && **str <= '9')
            ++(*str);
        
        return *str > before;
    }
    
    bool scanInteger(char** str)
    {
        if(**str == '+' || **str == '-')
            ++(*str);
        
        return scanUnsignedInteger(str);
    }

};