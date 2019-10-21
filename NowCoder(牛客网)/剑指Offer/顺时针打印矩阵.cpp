/*
**链接：https://www.nowcoder.com/questionTerminal/9b4c81a02cd34f76be2659fa0d54342a
**描述：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
**时间：2019.10.21
**思路：略
*/

class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> result;
        
        if(matrix.empty())
            return result;
        
        int borderTop, borderBottom, borderLeft, borderRight;
        borderTop = 0;
        borderBottom = matrix.size()-1;
        borderLeft = 0;
        borderRight = matrix[0].size()-1;
        
        int x = 0;
        int y = 0;
        
        while(true)
        {
            while(x <= borderRight)
            {
                result.push_back(matrix[y][x]);
                x++;
            }
            x--;
            y++;
            borderTop++;
            
            if(borderTop > borderBottom)
                break;
            
            while(y <= borderBottom)
            {
                result.push_back(matrix[y][x]);
                y++;
            }
            y--;
            x--;
            borderRight--;
            
            if(borderLeft > borderRight)
                break;
            
            while(x >= borderLeft)
            {
                result.push_back(matrix[y][x]);
                x--;
            }
            x++;
            y--;
            borderBottom--;
            
            if(borderBottom < borderTop)
                break;
            
            while(y >= borderTop)
            {
                result.push_back(matrix[y][x]);
                y--;
            }
            y++;
            x++;
            borderLeft++;
            
            if(borderLeft > borderRight)
                break;
        }
        
        return result;
    }
};