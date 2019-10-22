/*
**链接：http://bestcoder.hdu.edu.cn/contests/contest_showproblem.php?cid=861&pid=1005
**
Problem Description
度熊发现，1, 3 以及 9这三个数字很神奇，它们的所有的倍数的每位数字的和一定是自己的倍数。例如说: 54 是 3 的倍数，同时 5+4 =9也是3的倍数。在另一个例子666是9的倍数，同时6+6+6 =18也是9的倍数。

度熊又发现，除了 1, 3, 9以外的的正整数，虽然并不满足"所有的倍数的每位数字的和一定是自己的倍数"，但也存在一些数是它们的倍数且各位数字和也是它们的倍数。例如说，888 是 12 的倍数，且他的各位数字和 8+8+8=24 也是 12的倍数。

现在度熊想知道，给你一个正整数 V，是否存在一个数 x，使得 V 是 x的倍数，同时它的每位数字的和也是 x 的倍数呢?请找出所有这样的数 x。

Input
有多组询问，第一行包含一个正整数 TT 代表有几组询问，接着每组测试数据占一行，包含一个正整数 V。
1≤T≤100
1≤V≤10^9
​​
Output
对于每一个询问，输出两行，第一行包含一个正整数 m，m 代表对于该询问的 V，有几个满足条件的 x。第二行输出 m 个数，把所有满足条件的 x 由小到大输出。
**时间：2019.8.18
**思路：略
*/

#include <iostream>
using namespace std;

int main()
{
    int T;
    cin >> T;

    long long n;
    for (int i = 0; i < T; i++)
    {
        cin >> n;

        int sum = 0;
        long long t = n;
        while (true)
        {
            sum += (t % 10);
            if (t < 10)
                break;
            t /= 10;
        }

        bool first = true;
        long long count = 0;

        for (int k = 1; k <= sum; k++)
            if (sum%k == 0 && n%k == 0)
                count++;

        cout << count << endl;

        for (int k = 1; k<=sum; k++)
        {
            if (sum%k == 0 && n%k == 0)
            {
                if (first)
                {
                    cout << k;
                    first = false;
                }
                else
                    cout << " " << k;
            }
        }
        cout << endl;
     }
    return 0;
}
