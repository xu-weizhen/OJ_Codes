
def complemented(s):
    res = ''
    for ch in s:
        if ch == '0':
            res += '1'
        elif ch == '1':
            res += '0'
        else:
            res += ch
    return res

def reversed(s):
    return s[::-1]


def comAndRev(s):
    return reversed(complemented(s))

inputs = input().split(' ')
T = inputs[0]
B = inputs[1]

for t in range(T):
    finish = 0
    bits = '?' * B

    sure = False
    judge = ''
    judge_bit = 1


    count = 0
    while count < 150:
        if count % 10 == 1:
            sure = False
            judge = ''
            judge_bit = 1

        if not sure and finish < 2:
            print(judge_bit)
            ans = input()
            judge += ans
            
            c = complemented(bits)[:len(judge)]
            r = reversed(bits)[:len(judge)]
            cr = comAndRev(bits)[:len(judge)]

            if judge == c and judge != r and judge != cr:
                sure = True
                bits = complemented(bits)
            elif judge != c and judge == r and judge != cr:
                sure = True
                bits = reversed(bits)
            elif judge != c and judge != r and judge == cr:
                sure = True
                bits = comAndRev(bits)
            
            judge_bit += 1
        else:
            if finish < B:
                for i in range(len(bits)):
                    if bits[i] == '?':
                        break
                
                if i < len(bits):
                    print(i)
                    ans = input()
                    
                    if i < len(bits) - 1:
                        bits = bits[:i] + ans + bits[i + 1:]
                    elif i == len(bits) - 1:
                        bits = bits[:i] + ans
            else:
                print(1)
                ans = input()
        
        count += 1
    
    print(bits)
    ans = input()
    if ans == 'N':
        break















