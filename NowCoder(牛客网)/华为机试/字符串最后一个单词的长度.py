#
#链接：https://www.nowcoder.com/questionTerminal/8c949ea5f36f422594b306a2300315da
#描述：计算字符串最后一个单词的长度，单词以空格隔开。
#时间：2019.10.13
#思路：略
#

string = input()
str_list = string.split()
length = len(str_list)
print(len(str_list[length-1]))