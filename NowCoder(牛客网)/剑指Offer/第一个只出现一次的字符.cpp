/*
**链接：https://www.nowcoder.com/questionTerminal/1c82e8cf713b4bbeb2a5b31cf5b0417c
**描述：在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
**时间：2019.8.8
**思路：略
*/

class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        bool *appear = (bool*)malloc(sizeof(bool)*str.length());
        for(int i=0; i<str.length(); i++)
            appear[i] = false;
        
        int index = 0;
        int find;
        bool f;
        while(index<str.length())
        {
            if(appear[index])
            {
                index++;
                continue;
            }
            
            find = index +1;
            f = true;
            while(find<str.length())
            {
                if(str[index]==str[find])
                {
                    f = false;
                    appear[find] = true;
                }
                find++;
            }
            if(f)
                break;
            appear[index] = true;
            index++;
        }
        
        free(appear);
        
        if(f)
            return index;
        else
            return -1;
    }
};

// 解法二
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        if(str.empty())
            return -1;
        
        int* table = new int[256];
        for(int i = 0; i < 256; i++)
            table[i] = -1;
        
        for(int i = 0; str[i] != '\0'; i++)
        {
            if(table[str[i]] == -1)
                table[str[i]] = i;
            else
                table[str[i]] = -2;
        }
        
        int min = 260;
        for(int i = 0; i < 256; i++)
        {
            if(table[i] >= 0 && table[i] < min)
                min = table[i];
        }
        
        return min;
    }
};