/*
**链接：https://www.nowcoder.com/questionTerminal/4c776177d2c04c2494f2555c9fcc1e49
**描述：定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
**时间：2019.8.8
**思路：略
*/

class Solution {
public:
    stack<int> s;
    stack<int> s_min;
    
    void push(int value) {
        if(s.empty())
        {
            s.push(value);
            s_min.push(value);
        }
        else
        {
            if(value <= s_min.top())
            {
                s.push(value);
                s_min.push(value);
            }
            else
                s.push(value);
        }
    }
    void pop() {
        if(s.top() == s_min.top())
        {
            s.pop();
            s_min.pop();
        }
        else
            s.pop();
    }
    int top() {
        return s.top();
    }
    int min() {
        return s_min.top();
    }
};