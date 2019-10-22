/*
**链接：http://bestcoder.hdu.edu.cn/contests/contest_showproblem.php?cid=861&pid=1001
**
Problem Description
度度熊最近学习了多项式和极限的概念。 现在他有两个多项式 f(x)f(x) 和 g(x)g(x)，他想知道当 xx 趋近无限大的时候，f(x) / g(x)f(x)/g(x) 收敛于多少。

Input
第一行一个整数 T(1 \leq T \leq 100)T (1≤T≤100) 表示数据组数。 对于每组数据，第一行一个整数 n(1 \leq n \leq 1,000)n (1≤n≤1,000)，n-1n−1 表示多项式 ff 和 gg 可能的最高项的次数（最高项系数不一定非0）。 接下来一行 nn 个数表示多项式 ff，第 ii 个整数 f_i(0 \leq f_i \leq 1,000,000)f
​i
​​  (0≤f
​i
​​ ≤1,000,000) 表示次数为 i-1i−1 次的项的系数。 接下来一行 nn 个数表示多项式 gg，第 ii 个整数 g_i(0 \leq g_i \leq 1,000,000)g
​i
​​  (0≤g
​i
​​ ≤1,000,000) 表示次数为 i-1i−1 次的项的系数。 数据保证多项式 ff 和 gg 的系数中至少有一项非0。

Output
对于每组数据，输出一个最简分数 a/ba/b（aa 和 bb 的最大公约数为1）表示答案。 如果不收敛，输出 1/01/0。
**时间：2019.8.17
**思路：略
*/

#include <iostream>
using namespace std;


int main()
{
    int T;
    int n;
    int *p1, *p2;
    int now;
    bool f = false;

    cin >> T;

    for (int i = 0; i < T; i++)
    {
        cin >> n;

        p1 = (int*)malloc(sizeof(int)*n);
        p2 = (int*)malloc(sizeof(int)*n);
        f = false;
        now = n - 1;

        for (int i = 0; i < n; i++)
            cin >> p1[i];

        for (int i = 0; i < n; i++)
            cin >> p2[i];

        while (!f)
        {
            if (p1[now] != 0 && p2[now] != 0)
            {
                int temp, r;
                int a = p1[now];
                int b = p2[now];
                if (a < b)
                {
                    temp = a;
                    a = b;
                    b = temp;
                }
                while (b != 0)
                {
                    r = a%b;
                    a = b;
                    b = r;
                }

                cout << p1[now]/a << "/" << p2[now]/a << endl;
                f = true;
            }

            if (p1[now] == 0 && p2[now] != 0)
            {
                cout << "0/1" << endl;
                f = true;
            }

            if (p1[now] != 0 && p2[now] == 0)
            {
                cout << "1/0" << endl;
                f = true;
            }

            now--;
        }
    }
    return 0;
}
