'''
√ × ×
'''
T = int(input())

for t in range(T):
    n = int(input())

    pa = []
    l = []  # 当前下标

    for i in range(n):
        p = input()
        pa.append(p)
        l.append(len(p) - 1)

    res = ''
    march = True
    change = True

    while True:
        while change and march:
            change = False
            ch = ''
            for index in range(len(pa)):
                if l[index] >= 0:
                    
                    if l[index] == 0 and pa[index][0] == '*':
                        l[index] -= 1
                        change = True
                        continue
                    
                    if not ch:
                        if pa[index][l[index]]== '*':
                            if res != '' and l[index] - 1 >= 0 and pa[index][l[index] - 1] == res[0]:
                                l[index] -= 2
                                change = True
                                if l[index] >= 0 and pa[index][l[index]] != '*':
                                    ch = pa[index][l[index]]
                                    l[index] -= 1
                                continue
                            else:
                                pass
                        else:
                            ch = pa[index][l[index]]
                            l[index] -= 1
                            change = True
                    else:
                        change = True
                        if pa[index][l[index]] == ch:
                            l[index] -= 1
                        elif pa[index][l[index]] == '*' and l[index] - 1 >= 0 and pa[index][l[index] - 1] == ch:
                            l[index] -= 2
                        elif pa[index][l[index]] == '*':
                            pass
                        else:
                            if l[index] + 1 < len(pa[index]) and pa[index][l[index] + 1] == '*':
                                l[index] += 1
                            elif l[index] + 2 < len(pa[index]) and pa[index][l[index] + 2] == '*':
                                l[index] += 2
                            else:
                                march = False
                                break
            res = ch + res 
        
        if not march:
            break

        leave = False
        for i in range(len(l)):
            if l[i] > 0:
                l[i] -= 1
                leave = True
                change = True
                break
        
        if not leave:
            break

    if march:
        print('Case #{}: {}'.format(t + 1, res))
    else:
        print('Case #{}: *'.format(t + 1))


'''
Pattern Matching

Problem
Many terminals use asterisks (*) to signify "any string", including the empty string. For example, when listing files matching BASH*, a terminal may list BASH, BASHER and BASHFUL. For *FUL, it may list BEAUTIFUL, AWFUL and BASHFUL. When listing B*L, BASHFUL, BEAUTIFUL and BULL may be listed.

In this problem, formally, a pattern is a string consisting of only uppercase English letters and asterisks (*), and a name is a string consisting of only uppercase English letters. A pattern p matches a name m if there is a way of replacing every asterisk in p with a (possibly empty) string to obtain m. Notice that each asterisk may be replaced by a different string.

Given N patterns, can you find a single name of at most 104 letters that matches all those patterns at once, or report that it cannot be done?

Input
The first line of the input gives the number of test cases, T. T test cases follow. Each test case starts with a line with a single integer N: the number of patterns to simultaneously match. Then, N lines follow, each one containing a single string Pi representing the i-th pattern.

Output
For each test case, output one line containing Case #x: y, where x is the test case number (starting from 1) and y is any name containing at most 104 letters such that each Pi matches y according to the definition above, or * (i.e., just an asterisk) if there is no such name.

Limits
Time limit: 20 seconds per test set.
Memory limit: 1GB.
1 ≤ T ≤ 100.
2 ≤ N ≤ 50.
2 ≤ length of Pi ≤ 100, for all i.
Each character of Pi is either an uppercase English letter or an asterisk (*), for all i.
At least one character of Pi is an uppercase English letter, for all i.

Test set 1 (Visible Verdict)
Exactly one character of Pi is an asterisk (*), for all i.
The leftmost character of Pi is the only asterisk (*), for all i.

Test set 2 (Visible Verdict)
Exactly one character of Pi is an asterisk (*), for all i.

Test set 3 (Visible Verdict)
At least one character of Pi is an asterisk (*), for all i.

Sample

Input
 	
Output
 
2
5
*CONUTS
*COCONUTS
*OCONUTS
*CONUTS
*S
2
*XZ
*XYZ

  
Case #1: COCONUTS
Case #2: *

  
In Sample Case #1, there are other possible answers, including COCOCONUTS and ILIKECOCONUTS. Neither COCONUTSAREGREAT nor COCOANUTS would be acceptable. Notice that the same pattern may appear more than once within a test case.

In Sample Case #2, there is no acceptable name, so the answer is *.

The following cases could not appear in Test Set 1, but could appear in Test Set 2 or Test Set 3:

  4
  H*O
  HELLO*
  *HELLO
  HE*
HELLO and HELLOGOODBYEHELLO are examples of acceptable answers. OTHELLO and HELLOO would not be acceptable.

  2
  CO*DE
  J*AM
There is no name that matches both patterns, so the answer would be *.

  2
  CODE*
  *JAM
CODEJAM is one example of an acceptable answer.

The following cases could not appear in Test Set 1 or Test Set 2, but could appear in Test Set 3:

  2
  A*C*E
  *B*D*
ABCDE and ABUNDANCE are among the possible acceptable answers, but BOLDFACE is not.

  2
  A*C*E
  *B*D
There is no name that matches both patterns, so the answer would be *.

  2
  **Q**
  *A*
QUAIL and AQ are among the possible acceptable answers here.
'''
