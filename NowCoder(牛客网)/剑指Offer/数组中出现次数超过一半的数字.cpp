/*
**链接：https://www.nowcoder.com/questionTerminal/e8a1b01a2df14cb2b228b30ee6a92163
**描述：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
**时间：2019.10.29
**思路：略
*/

class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.empty())
            return 0;
        
        int num;
        int count;
        vector<int>::iterator ite = numbers.begin();
        num = *ite;
        count = 1;
        
        for(; ite != numbers.end(); ite++)
        {
            if(*ite == num)
                count++;
            else
            {
                count--;
                if(count == 0)
                {
                    num = *ite;
                    count = 1;
                }
            }
        }
        
        count = 0;
        for(ite = numbers.begin(); ite != numbers.end(); ite++)
            if(*ite == num)
                count++;
        
        int len = numbers.size();
        if(count > len / 2)
            return num;
        else
            return 0;
    }
};