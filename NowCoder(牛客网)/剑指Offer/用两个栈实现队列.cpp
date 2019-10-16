/*
**链接：https://www.nowcoder.com/questionTerminal/54275ddae22f475981afa2244dd448c6
**描述：用两个栈来实现一个队列，完成队列的Push和Pop操作。队列中的元素为int类型。
**时间：2019.10.14
**思路：略
*/

class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        if(stack2.empty())
        {
            int t;
            while(!stack1.empty())
            {
                t = stack1.top();
                stack1.pop();
                stack2.push(t);
            }
        }
        
        //if(stack2.empty())
        //    throw new exception("Queue is empty");
        
        int t = stack2.top();
        stack2.pop();
        return t;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};