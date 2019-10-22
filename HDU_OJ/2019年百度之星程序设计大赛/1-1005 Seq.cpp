/*
**链接：http://bestcoder.hdu.edu.cn/contests/contest_showproblem.php?cid=861&pid=1005
**
Problem Description
度度熊有一个递推式 a_{n} = (\sum_{i=1}^{n-1} a_{i}*i) % n 其中 a_1 = 1。现给出 n，需要求 a_n。

Input
第一行输入一个整数 T，代表 T(1≤T≤100000) 组数据。 接下 T 行，每行一个数字 n(1≤n≤10^12)。

Output
输出 T行，每行一个整数表示答案。
**时间：2019.8.17
**思路：略
*/

#include <iostream>
using namespace std;

int main()
{
    int T;
    cin >> T;

    long long in;
    long long a;
    long long b;
    for (int i = 0; i < T; i++)
    {
        cin >> in;
        
        if (in == 1 || in == 2)
            cout << "1" << endl;
        else
        {
            a = (in - 3) / 6;
            b = (in - 3) % 6;

            switch (b)
            {
            case(0):
            case(2):
                cout << a << endl;
                break;
            case(1):
                cout << 3 + a * 6 << endl;
                break;
            case(3):
                cout << 3 + a * 3 << endl;
                break;
            case(4):
                cout << 5 + a * 4 << endl;
                break;
            case(5):
                cout << 4 + a * 3 << endl;
                break;
            }
        }
    }

    return 0;
}
