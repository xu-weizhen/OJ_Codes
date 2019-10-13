/*
**链接：https://www.nowcoder.com/questionTerminal/3e483fe3c0bb447bb17ffb3eeeca78ba
**描述：计算给定3个数a，b，c，在它们中间添加"+"， "*"， "("， ")"符号，能够获得的最大值。
**时间：2019.10.4
**思路：略
*/

#include <iostream>
using namespace std;

int main()
{
    int a, b, c;
    int result;
    int temp;
    
    cin >> a >> b >> c;
    
    //a+b+c
    result = a+b+c;
    
    //a*b*c
    temp = a*b*c;
    if(temp > result)
        result = temp;
    
    //(a+b)*c
    temp = (a+b)*c;
    if(temp > result)
        result = temp;
    
    //a*(b+c)
    temp = a*(b+c);
    if(temp > result)
        result = temp;
    
    //a+b*c
    temp = a+b*c;
    if(temp > result)
        result = temp;
    
    //a*b+c
    temp = a*b+c;
    if(temp > result)
        result = temp;
    
    cout << result << endl;
    return 0;
}

//另解
/*
#include <iostream>
using namespace std;

int main()
{
    int a, b, c;
    int result;
    int temp;
    
    cin >> a >> b >> c;
    
    if(b>a && b>c)
    {
        if(a==1)
            result = (a+b)*c;
        else if(c==1)
            result = a*(b+c);
        else
            result = a*b*c;
    }
    else if(a>b && a>c)
    {
        if(b+c > b*c)
            result = a*(b+c);
        else
            result = a*(b*c);
    }
    else
    {
        if(a+b > a*b)
            result = c*(a+b);
        else
            result = c*(a*b);
    }
    cout<<result<<endl;
    return 0;
}
*/