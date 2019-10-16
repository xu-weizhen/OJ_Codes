/*
**链接：https://www.nowcoder.com/questionTerminal/8c949ea5f36f422594b306a2300315da
**描述：计算字符串最后一个单词的长度，单词以空格隔开。
**时间：2019.10.13
**思路：略
*/

#include <iostream>
#include <string.h>
using namespace std;

int main()
{
    char input[5001];
    cin.getline(input, 5001);
    int length = strlen(input);
    int result = 0;
    for(int i=length-1; i>=0 ; i--)
    {
        if(input[i]!=' ')
            result++;
        else
            break;
    }
    cout << result << endl;
    return 0;
}