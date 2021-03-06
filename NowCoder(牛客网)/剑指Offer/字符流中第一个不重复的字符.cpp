/*
**链接：https://www.nowcoder.com/questionTerminal/00de97733b8e4f97a3fb5c680ee10720
**描述：请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
**时间：2019.10.30
**思路：略
*/
class Solution
{
public:
    int occurrence[256];
    int index;
    
    Solution()
    {
        for(int i = 0; i < 256; ++i)
            occurrence[i] = -1;
        index = 0;
    }
    
    //Insert one char from stringstream
    void Insert(char ch)
    {
         if(occurrence[ch] == -1)
            occurrence[ch] = index;
        else if(occurrence[ch] >= 0)
            occurrence[ch] = -2;
        index++;
    }
    //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        char ch = '#';
        int minIndex = numeric_limits<int>::max();
        for(int i = 0; i < 256; ++i)
        {
            if(occurrence[i] >= 0 && occurrence[i] < minIndex)
            {
                ch = (char) i;
                minIndex = occurrence[i];
            }
        }
        return ch;
    }

};