/*
**链接：https://www.nowcoder.com/questionTerminal/d77d11405cc7470d82554cb392585106
**描述：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
**时间：2019.10.18
**思路：略
*/

class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        vector<int>::iterator push_element = pushV.begin();
        vector<int>::iterator pop_element = popV.begin();
        stack<int> s;
        bool result = false;
        
        for(; pop_element != popV.end(); pop_element++)
        {
            if(!s.empty() && s.top() == *pop_element)
                s.pop();
            else if(push_element == pushV.end())
                break;
            else if(*push_element == *pop_element)
                push_element++;
            else 
            {
                while(push_element != pushV.end() && *push_element != *pop_element)
                {
                    s.push(*push_element);
                    push_element++;
                }
                push_element++;
            }
        }
        if(s.empty() && pop_element == popV.end() && push_element == pushV.end())
            result = true;
        
        return result;
    }
};