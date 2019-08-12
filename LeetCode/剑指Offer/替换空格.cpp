/*
**链接：https://www.nowcoder.com/questionTerminal/4060ac7e3e404ad1a894ef3e17650423
**描述：请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
**时间：2019.8.12
**思路：略
*/

class Solution {
public:
	void replaceSpace(char *str,int length) {
        char *t = (char*)malloc(sizeof(char)*length);

        for (int i = 0; i<length; i++)
            t[i] = str[i];

        int i = 0;
        int j = 0;
        for (; i<length; i++)
        {
            if (t[i] != ' ')
                str[j++] = t[i];
            else
            {
                str[j++] = '%';
                str[j++] = '2';
                str[j++] = '0';
            }
        }
        str[j] = '\0';
        free(t);
	}
};