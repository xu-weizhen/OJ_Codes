
//题目：http://acm.hdu.edu.cn/showproblem.php?pid=1012
//时间：2019.12.7
//思路：略


#include <stdio.h>
#include <iostream>
#include <iomanip>
using namespace std;

int main() {
    cout << "n e" <<endl;
	cout << "- -----------"<<endl;
	long mul[10];
	mul[0] = 1;
	mul[1] = 1;
	for(int i = 2; i<10; i++)
	    mul[i] = mul[i-1] * i;
	
	double e[10];
	e[0] = 1;
	
	for(int n=1; n<10; n++)
	{
	    e[n] = e[n-1] + 1.0/mul[n];
	}
	
	for(int i=0; i<10; i++)
	{
	   if(i >= 3)
	        cout << i << " " << fixed << setprecision(9) <<e[i] << endl;
	   else
	        cout << i << " " << e[i] << endl;
	}
	return 0;
}