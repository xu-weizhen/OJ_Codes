/*
**链接：https://www.nowcoder.com/questionTerminal/6aa9e04fc3794f68acf8778237ba065b
**描述：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
**时间：2019.10.30
**思路：略
*/

class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index <= 0)
            return 0;
        
        int *num = new int[index];
        num[0] = 1;
        int next = 1;
        
        int multiply2 = 0;
        int multiply3 = 0;
        int multiply5 = 0;
        
        while(next < index)
        {
            int min = Min(num[multiply2] * 2, num[multiply3] * 3, num[multiply5] * 5);
            num[next] = min;
            
            while(num[multiply2] * 2 <= num[next])
                ++multiply2;
            while(num[multiply3] * 3 <= num[next])
                ++multiply3;
            while(num[multiply5] * 5 <= num[next])
                ++multiply5;
            
            ++next;
        }
        
        int ugly = num[next - 1];
        delete[] num;
        return ugly;
    }
    
    int Min(int number1, int number2, int number3)
    {
        int min = (number1 < number2) ? number1 : number2;
        min = (min < number3) ? min : number3;
        return min;
    }
};