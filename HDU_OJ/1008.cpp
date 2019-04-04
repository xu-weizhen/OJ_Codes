
//链接：http://acm.hdu.edu.cn/showproblem.php?pid=1008
//题目：Elevator
//时间：2017.2.16
//思路：略


#include <iostream>

using namespace std ;

int main()
{
    int a[101];
    int T ;
    cin >> T ;
    while (T!=0)
    {
        long long time=0 ;
        for (int i=1; i<=T; i++)
            cin>>a[i] ;
		
        a[0]=0;
        for (int i=1; i<=T; i++)
        {
            int c=a[i]-a[i-1] ;
            if ( c>0 )
            {
                time=time+c*6;
            } else {
                time=time+(-c)*4;
            }
        }
        cout << time+5*T << endl;
        cin >> T;
    }
}