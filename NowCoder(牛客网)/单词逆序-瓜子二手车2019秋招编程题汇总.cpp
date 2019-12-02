// 2019.12.2
/*
对于一个字符串，请设计一个算法，只在字符串的单词间做逆序调整，也就是说，字符串由一些由空格分隔的部分组成，你需要将这些部分逆序。
给定一个原字符串A，请返回逆序后的字符串。例，输入"I am a boy!", 输出"boy! a am I"

输入描述:
输入一行字符串str。(1<=strlen(str)<=10000)

输出描述:
返回逆序后的字符串。

输入例子1:
It's a dog!

输出例子1:
dog! a It's
*/

#include <iostream>

void reverse(char* start, char *end)
{
    while(start < end)
    {
        char t = *start;
        *start = *end;
        *end = t;
        
        start++;
        end--;
    }
}

int main()
{
    char str[10000];
    std::cin.getline(str, 10000);
    char *start = str;
    char *end = str;
    while(*end != '\0')
        end++;
    end--;
    
    reverse(start, end);
    
    start = str;
    end = str;
    while(true)
    {
        while(*end != ' ' && *end != '\0')
            end++;
        end--;
        reverse(start, end);
        end++;
        if(*end == '\0')
            break;
        start = ++end;
    }
    
    std::cout << str << std::endl;
    
    return 0;
}