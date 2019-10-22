/*
**链接：https://www.nowcoder.com/questionTerminal/a861533d45854474ac791d90e447bafd
**描述：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
**时间：2019.10.22
**思路：略
*/

class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        if(sequence.empty())
            return false;
        else
            return check(sequence);
    }
    
    bool check(vector<int> sequence)
    {
        if(sequence.size()==1 || sequence.empty())
            return true;
        
        int count = sequence.size();
        int left = 0;
        while(sequence[left] < sequence[count-1])
            left++;
        int right = left;
        while(right < count-1)
        {
            if(sequence[right] < sequence[count-1])
                return false;
            right++;
        }
        
        vector<int> leftSeq;
        vector<int> rightSeq;
        for(int i=0; i<left; i++)
            leftSeq.push_back(sequence[i]);
        for(int i=left; i<count-1; i++)
            rightSeq.push_back(sequence[i]);
        
        return check(leftSeq) && check(rightSeq);
    }
};