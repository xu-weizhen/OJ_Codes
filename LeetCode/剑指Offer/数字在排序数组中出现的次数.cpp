/*
**链接：https://www.nowcoder.com/questionTerminal/70610bf967994b22bb1c26f9ae901fa2
**描述：统计一个数字在排序数组中出现的次数
**时间：2019.10.4
**思路：先二分查找到该数值，在向该位置前后计数出现次数
*/

class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int front = 0;
        int rear = data.size() - 1;
        int mid;

        while (front <= rear)
        {
            if (data[front] == k)
            {
                mid = front;
                break;
            }
            mid = (front + rear) / 2;

            if (data[mid] > k)
                rear = mid - 1;
            else
                if (data[mid] < k)
                    front = mid + 1;
                else
                    break;
        }

        int forward = 1;
        int backward = 0;
        int count = 0;
        while (mid - forward >= 0)
        {
            if (data[mid - forward] == k)
            {
                count++;
                forward++;
            }
            else
                break;
        }
        while (mid + backward < data.size())
        {
            if (data[mid + backward] == k)
            {
                count++;
                backward++;
            }
            else
                break;
        }
        return count;
    }
};

//另解
/*
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        auto l = lower_bound(data.begin(), data.end(), k);
        auto r = upper_bound(data.begin(), data.end(), k);
        return r - l;
    }
};
*/