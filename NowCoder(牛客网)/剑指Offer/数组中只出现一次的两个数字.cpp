/*
**链接：https://www.nowcoder.com/questionTerminal/e02fdb54d7524710a7d664d082bb7811
**描述：一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
**时间：2019.10.31
**思路：一个数异或自身为0，将所有数进行异或运算，依据结果中不为0的位，将数据分为两组，再在组内进行异或运算
*/

class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        if(data.size() < 2)
            return;
        
        int allOr = 0;
        vector<int>::iterator iter = data.begin();
        for(; iter != data.end(); iter++)
            allOr ^= *iter;
        
        unsigned int index = 0;
        while((allOr & 1) == 0 && (index < 8 * sizeof(int)))
        {
            allOr = allOr >> 1;
            ++index;
        }
        
        *num1 = 0;
        *num2 = 0;
        for(iter = data.begin(); iter != data.end(); iter++)
        {
            if(IsBit(*iter, index))
                *num1 ^= *iter;
            else
                *num2 ^= *iter;
        }
    }
    
    bool IsBit(int num, int index)
    {
        num = num >> index;
        return (num & 1);
    }
};