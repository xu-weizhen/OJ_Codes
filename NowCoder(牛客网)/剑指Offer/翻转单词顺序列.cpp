/*
**链接：https://www.nowcoder.com/questionTerminal/3194a4f4cf814f63919d0790578d51f3
**描述：牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
**时间：2019.10.31
**思路：略
*/

class Solution {
public:
    string ReverseSentence(string str) {
        if(str.size() <= 1)
            return str;
        
        string result = Rev(str);
        int front = 0;
        int rear = 0;
        
        while(rear < result.size())
        {
            while(result[rear] != ' ' && rear < result.size())
                rear++;
            
            int tf = front;
            int tr = rear - 1;
            while(tf < tr)
            {
                char tc;
                tc = result[tf];
                result[tf] = result[tr];
                result[tr] = tc;
                tf++;
                tr--;
            }
            
            front = rear + 1;
            rear++;
        }
        
        return result;
    }
    
    string Rev(string str)
    {
        string s(str.rbegin(),str.rend());
        return s;
    }
};