/*
**链接：https://www.nowcoder.com/questionTerminal/fe6b651b66ae47d7acce78ffdd9a96c7
**描述：输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
**时间：2019.10.29
**思路：略
*/

class Solution {
public:
    vector<string> result;
    vector<string> Permutation(string str) {
        result.clear();
        
        if(str.empty())
            return result;
        
        Permutation(str, 0);
        sort(result.begin(), result.end());
        return result;
    }
    
    void Permutation(string str, int begin)
    {
        if(str[begin] == '\0')
        {
            vector<string>::iterator it;
            it = find(result.begin(), result.end(), str);
            if(it == result.end())
                result.push_back(str);
        }
        else
        {
            for(int ch = begin; str[ch] != '\0'; ch++)
            {
                char temp = str[ch];
                str[ch] = str[begin];
                str[begin] = temp;
                
                Permutation(str, begin + 1);
                
                temp = str[ch];
                str[ch] = str[begin];
                str[begin] = temp;
            }
        }
    }
};