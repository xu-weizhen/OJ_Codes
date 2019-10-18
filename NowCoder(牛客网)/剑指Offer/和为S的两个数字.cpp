/*
**链接：https://www.nowcoder.com/questionTerminal/390da4f7a00f44bea7c2f3d19491311b
**描述：输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
**时间：2019.10.18
**思路：略
*/

class Solution {
public:
    struct result
    {
        int num1;
        int num2;
    };
    
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        int length = array.size();
        int front = 0;
        int rear = length - 1;
        result r;
        bool first = true;
        while(front < rear)
        {
            while(array[front] + array[rear] < sum && front < rear)
                front++;
            if(array[front] + array[rear] == sum)
            {
                if(first)
                {
                    r.num1 = array[front];
                    r.num2 = array[rear];
                    first = false;
                }
                else
                {
                    if(array[front] * array[rear] < r.num1 * r.num2)
                    {
                        r.num1 = array[front];
                        r.num2 = array[rear];
                    }
                }
                front++;
            }
            
            while(array[front] + array[rear] > sum && front < rear)
                rear--;
            if(array[front] + array[rear] == sum)
            {
                if(first)
                {
                    r.num1 = array[front];
                    r.num2 = array[rear];
                    first = false;
                }
                else
                {
                    if(array[front] * array[rear] < r.num1 * r.num2)
                    {
                        r.num1 = array[front];
                        r.num2 = array[rear];
                    }
                }
                rear--;
            }
        }
        vector<int> v;
        if(first)
            return v;
        v.push_back(r.num1);
        v.push_back(r.num2);
        return v;
    }
};