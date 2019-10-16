/*
**链接：https://www.nowcoder.com/questionTerminal/a35ce98431874e3a820dbe4b2d0508b1
**描述：写出一个程序，接受一个由字母和数字组成的字符串，和一个字符，然后输出输入字符串中含有该字符的个数。不区分大小写。
**时间：2019.10.13
**思路：略
*/

#include <iostream>
#include <string>
using namespace std;

int main()
{
    string str;
    getline(cin, str);
    char target;
    cin >> target;
    int count = 0;
    for(int i=0; str[i]!='\0'; i++)
    {
        if(str[i]==target || str[i]==char(target-32) || str[i]==char(target+32))
            count++;
    }
    cout << count << endl;
    return 0;
}
