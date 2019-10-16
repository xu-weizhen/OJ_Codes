/*
**链接：https://www.nowcoder.com/questionTerminal/beb5aa231adc45b2a5dcc5b62c93f593
**描述：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
**时间：2019.10.14
**思路：略
*/

class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int size = array.size();
       
        if(size<=1)
            return;
        
        for(int i=0; i<size; i++)
        {
            if(array[i]%2!=0)
                q1.push(array[i]);
            else
                q2.push(array[i]);
        }
        
        int p = 0;
        while(!q1.empty())
        {
            array[p] =  q1.front();
            q1.pop();
            p++;
        }
        
        while(!q2.empty())
        {
            array[p] =  q2.front();
            q2.pop();
            p++;
        }
        
    }
private:
    queue<int> q1;
    queue<int> q2;
};