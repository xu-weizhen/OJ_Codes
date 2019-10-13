/*
**链接：https://www.nowcoder.com/questionTerminal/abc3fe2ce8e146608e868a70efebf62e
**描述：在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
**时间：2019.10.13
**思路：数组右上角开始查找；若该数小于目标，看下一行；若该数大于目标，看上一列
*/

class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        int row = 0;
        int col = array[0].size()-1;
        bool find = false;
        while(row<array.size() && col>=0)
        {
            if(target == array[row][col])
            {
                find = true;
                break;
            }
            else 
            {
                if(target < array[row][col])
                    col--;
                else
                    row++;
            }
        }
        return find;
    }
};