
//链接：http://acm.hdu.edu.cn/showproblem.php?pid=1001
//题目：Sum Problem
//时间：2017.2.15
//思路：略


#include <iostream>

using namespace std ;

int main()
{
	int a;
	long long sum=0 ;
	while(cin >> a)
	{
		for (int i=1; i<=a ; i++)
			sum+=i ;
		cout << sum << endl << endl;
		sum=0;
	}
}