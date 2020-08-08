
//问题：http://acm.hdu.edu.cn/showproblem.php?pid=2067
//时间：2020.8.8
//思路：动态规划

#include <iostream>
using namespace std;

int main()
{
    int count = 1;

    int n = 0;
    cin >> n;

    while(n != -1)
    {
        long long* arr = new long long[n + 1];

        for(int i=0; i<=n; i++)
            arr[i] = 1;
        
        for(int i=1; i<=n; i++)
        {
            for(int j=i + 1; j<=n; j++)
            {
                arr[j] = arr[j] + arr[j - 1];
            }
        }

        cout << count << " " << n << " " << arr[n] * 2 << endl;

        cin >> n;
        count ++;
    }

    return 0;
}
