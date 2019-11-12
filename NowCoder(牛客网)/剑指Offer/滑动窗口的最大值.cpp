/*
**链接：https://www.nowcoder.com/questionTerminal/1624bc35a45c42c0bc17d17fa0cba788
**描述：给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
**时间：2019.10.31
**思路：略
*/

class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        int length = num.size();
        vector<int> result;
        if(size == 0)
            return result;
        
        int front = 0;
        int rear = front + size -1;
        
        while(rear < length)
        {
            int t = front;
            int max = num[t];
            t++;
            while(t <= rear)
            {
                if(num[t] > max)
                    max = num[t];
                t++;
            }
            result.push_back(max);
            front++;
            rear++;
        }
        
        return result;
    }
};