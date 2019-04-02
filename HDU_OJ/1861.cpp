
//问题：http://acm.hdu.edu.cn/showproblem.php?pid=1861
//时间：2019.4.2
//思路：数组保存每艘船的被借次数，被借总时间和上次被借时间

#include "stdafx.h"

#include <iostream>
#include <string.h>
using namespace std;

typedef struct note
{
	int sum;		//总计被借时间
	int times;		//被借次数
	char begin[6];	//被借时刻
} boat;

int main()
{
	boat bs[101];
	
	//初始化
	for(int i=0; i<101; i++)
	{
		bs[i].sum = 0;
		bs[i].begin[0] = '\0';
		bs[i].times = 0;
	}

	int num;		//船编号
	char status;	//借/还
	char time[6];

	while(cin >> num)
	{
		if(num==-1)
			break;

		if(num==0)
		{
			int result1 = 0;		//被借次数
			int result2 = 0;		//平均时间

			for(int i=0; i<101; i++)
			{
				result1+=bs[i].times;
				bs[i].times = 0;

				result2+=bs[i].sum;
				bs[i].sum = 0;
			}

			cout << result1 << " " ;
			if(result1==0)
				cout << "0" <<endl;
			else
				cout << int((float)result2/result1+0.5) << endl;
			cin >> status >> time;
			continue;
		}

		cin >> status >> time;
		if(status=='S')
		{
			strcpy(bs[num].begin, time);		//记录被借时间
		}
		else
		{	
			if(bs[num].begin[0]!='\0')
			{
				int ehour = (time[0]-'0')*10+(time[1]-'0');
				int emin = (time[3]-'0')*10+(time[4]-'0');
				int shour = (bs[num].begin[0]-'0')*10+(bs[num].begin[1]-'0');
				int smin = (bs[num].begin[3]-'0')*10+(bs[num].begin[4]-'0');
			
				bs[num].times++;				//被借次数
				bs[num].sum += ((ehour*60+emin) - (shour*60+smin));		//增加被借时间
				bs[num].begin[0] = '\0';		//初始化被借时间
			}
		}
	}

	return 0;
}

