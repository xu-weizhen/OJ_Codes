/*
**链接：https://www.nowcoder.com/questionTerminal/6a296eb82cf844ca8539b57c23e6e9bf
**描述：输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
**时间：2019.10.29
**思路：略
*/

class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        multiset<int, greater<int>> leastNumbers;
        vector<int> result;
        
        if(k < 1 || input.size() < k)
            return result;
        
        vector<int>::const_iterator iter = input.begin();
        multiset<int, greater<int> >::iterator iterGreatest;
        
        for(; iter != input.end(); ++ iter)
        {
            if((leastNumbers.size()) < k)
                leastNumbers.insert(*iter);
            else
            {
                iterGreatest = leastNumbers.begin();
                if(*iter < *(leastNumbers.begin()))
                {
                    leastNumbers.erase(iterGreatest);
                    leastNumbers.insert(*iter);
                }
            }
        }
        
        int i = 0;
        iterGreatest = leastNumbers.begin();
        for(; iterGreatest != leastNumbers.end(); ++iterGreatest, ++i)
            result.push_back(*(iterGreatest));
        
        return result;
    }
};