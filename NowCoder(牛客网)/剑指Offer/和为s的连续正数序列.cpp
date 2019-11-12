/*
**链接：https://www.nowcoder.com/questionTerminal/c451a3fd84b64cb19485dad758a55ebe
**描述：小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck! 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
**时间：2019.10.31
**思路：略
*/

class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> result;
        vector<int> seq;
        
        if(sum < 3)
            return result;
        
        int small = 1;
        int big = 2;
        int middle = (sum + 1) / 2;
        int now = small + big;
        
        while(small < middle)
        {
            if(sum == now)
            {
                for(int i = small ; i <= big; i++)
                    seq.push_back(i);
                result.push_back(seq);
                seq.clear();
            }
            
            while(now > sum && small < middle)
            {
                now -= small;
                small++;
                
                if(sum == now)
                {
                    for(int i = small ; i <= big; i++)
                        seq.push_back(i);
                    result.push_back(seq);
                    seq.clear();
                }
            }
            
            big++;
            now += big;
        }
        return result;
    }
};