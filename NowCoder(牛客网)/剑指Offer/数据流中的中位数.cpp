/*
**链接：https://www.nowcoder.com/questionTerminal/9be0172896bd43948f8a32fb954e1be1
**描述：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
**时间：2019.10.29
**思路：略
*/

class Solution {
public:
    vector<int> min;
    vector<int> max;
    void Insert(int num)
    {
        if(((min.size() + max.size()) & 1) == 0)
        {
            if(max.size() > 0 && num < max[0])
            {
                max.push_back(num);
                push_heap(max.begin(), max.end(), less<int>());
                num = max[0];
                pop_heap(max.begin(), max.end(), less<int>());
                max.pop_back();
            }
            min.push_back(num);
            push_heap(min.begin(), min.end(), greater<int>());
        }
        else
        {
            if(min.size() > 0 && min[0] < num)
            {
                min.push_back(num);
                push_heap(min.begin(), min.end(), greater<int>());
                num = min[0];
                pop_heap(min.begin(), min.end(), greater<int>());
                min.pop_back();
            }
            max.push_back(num);
            push_heap(max.begin(), max.end(), less<int>());
        }
    }

    double GetMedian()
    { 
        int size = min.size() + max.size();
        if(size == 0)
            return 0;

        double median = 0;
        if((size & 1) == 1)
            median = min[0];
        else
            median = (double)(min[0] + max[0]) / 2;

        return median;
    }

};