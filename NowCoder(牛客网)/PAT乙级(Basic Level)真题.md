# A+B和C (15)

时间限制 1000 ms 内存限制 32768 KB 代码长度限制 100 KB 判断程序 Standard 



**题目描述**
给定区间[-2的31次方, 2的31次方]内的3个整数A、B和C，请判断A+B是否大于C。



**输入描述:**
输入第1行给出正整数T(<=10)，是测试用例的个数。随后给出T组测试用例，每组占一行，顺序给出A、B和C。整数间以空格分隔。



**输出描述:**
对每组测试用例，在一行中输出“Case #X: true”如果A+B>C，否则输出“Case #X: false”，其中X是测试用例的编号（从1开始）。




**输入例子:**
```
4
1 2 3
2 3 4
2147483647 0 2147483646
0 -2147483648 -2147483647
```



**输出例子:**

```
Case #1: false
Case #2: true
Case #3: true
Case #4: false
```



**代码**

```python
T = int(input())
 
for i in range(T):
    l = input().split(' ')
    res = str(int(l[0]) + int(l[1]) > int(l[2])).lower()
    print('Case #{}: {}'.format(i + 1, res))
```



