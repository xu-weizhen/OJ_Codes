
//题目：http://acm.hdu.edu.cn/showproblem.php?pid=1859
//时间：2019.3.31
//思路：若输入的点落在当前选定的矩形外，则根据点的位置扩大矩形

#include <iostream>
using namespace std;

int main()
{
	int lx = 240;	//长方形左下角x坐标
	int ly = 240;	//长方形左下角y坐标
	int rx = -240;	//长方形右上角x坐标
	int ry = -240;	//长方形右上角y坐标
	int a, b;		//输入的值
	while(cin>>a>>b)
	{
		//连续两次输入"0 0"
		if(a==0 && b==0 && lx==240 && ly==240 && rx==-240 && ry==-240)
			break;

		//输入为"0 0"
		if(a==0 && b==0)
		{
			cout << lx << " " << ly << " " << rx << " " << ry <<endl;
			lx = ly = 240;
			rx = ry = -240;
			continue;
		}

		//点落在当前矩形外时
		if(a < lx)
			lx = a;
		if(a > rx)
			rx = a;
		if(b < ly)
			ly = b;
		if(b > ry)
			ry = b;
	}
	return 0;
}

