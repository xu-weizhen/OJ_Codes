/*
**链接：https://www.nowcoder.com/questionTerminal/94a4d381a68b47b7a8bed86f2975db46
**描述：给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
**时间：2019.11.4
**思路：略
*/

class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        vector<int> B;
        
        B.push_back(1);
        for(int i = 1; i < A.size(); i++)
            B.push_back(B[i - 1] * A[i - 1]);
        
        double t = 1;
        for(int i = A.size() - 2; i >= 0; i--)
        {
            t *= A[i + 1];
            B[i] *= t;
        }
        
        return B;
    }
};