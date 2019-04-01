
//问题：http://acm.hdu.edu.cn/showproblem.php?pid=1002
//时间：2017.2.21
//思路：使用数组保存加数和结果

#include <iostream>
#include <string>
using namespace std ;

int main ()
{
	string a, b ;			//加数
	int T ;					//Case数
	int sum[1000] ;			//结果
	for ( int i=0 ; i<1000 ; i++ )
		sum[i]=0 ;
	
	cin >> T ;
	for (int i=1; i<=T; i++ )
	{
		cin >> a >> b;
		int lena=a.length();
		int lenb=b.length();
		int lena2=lena-1, lenb2=lenb-1;
		int duan ,cha ;
		
		//找位数较少的加数的位数及两个加数位数之差
		if (lena>lenb) 
		{
			duan = lenb ;	
			cha = lena-lenb ;
		}
		else 
		{
			duan = lena ;
			cha = lenb-lena ;
		}

		int m , n ;
		int jin = 0 ;			//进位
		int s = 0 ;
		
		//相加
		for (int k=duan-1; k>=0; k--, s++, lena2--, lenb2--)
		{
			m = a[lena2]-'0' ;
			n = b[lenb2]-'0' ;
			sum[s] = (m+n+jin)%10;
			jin = (m+n+jin)/10 ;
		}
		
		//处理两个加数相差的位数部分
		if (lena>lenb) 
		{
			int j = cha-1 ;
			for (int k=0; k<cha; k++, s++, j-- )
			{
				sum[s] = (jin+(a[j]-'0'))%10 ;
				jin = (jin+(a[j]-'0'))/10 ;
			}
		}
		else 
		{
			int j = cha-1 ;
			for (int k=0; k<cha; k++, s++,j-- )
			{
				sum[s] = (jin+(b[j]-'0'))%10 ;
				jin = (jin+(b[j]-'0'))/10 ;
			}
		}
		
		//输出结果
		std::cout << "Case " << i << ":" << endl ;
		std::cout << a << " + " << b << " = " ;
		for (int k=999; k>=0; k-- )
		{
			if (sum[k]!=0)
			{
				for ( ; k>=0; k-- )
					std::cout<<sum[k] ; 
				break ;
			}
		}
		std::cout<<endl ;
		
		if (i<T)
			cout<<endl ;

		//清零
		for (int i=0; i<1000; i++)
			sum[i]  = 0;

	}
	return 0;
}

