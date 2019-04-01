
//问题：http://acm.hdu.edu.cn/showproblem.php?pid=1860
//时间：2019.4.1
//思路：对第二行文本的字符作为map的key，保存每个字符出现次数。遍历第一行文本并找出对应字符在map中的value


#include <iostream>
#include <map>
using namespace std;

int main()
{
	char target[10];			//第一行文本
	char text[90];				//第二行文本
	map<char, int> result;
	map<char, int>::iterator iter;
	while(cin.getline(target, 10))
	{
		if(target[0]=='#' && target[1]=='\0')
			break;

		cin.getline(text, 90);
		for(int i=0; text[i]!='\0'; i++)
		{
			iter = result.find(text[i]);
			if(iter==result.end())
				result.insert(pair<char, int>(text[i], 1));
			else
				result[text[i]]+=1;
		}

		for(int i=0; target[i]!='\0'; i++)
		{
			iter = result.find(target[i]);
			cout<<iter->first<<" "<<iter->second<<endl;
		}

		//清空
		for(iter=result.begin(); iter!= result.end();)
			result.erase(iter++);
	}

	return 0;
}