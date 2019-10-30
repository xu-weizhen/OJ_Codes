/*
**链接：https://www.nowcoder.com/questionTerminal/8b3b95850edb4115918ecebdf1b4d222
**描述：输入一棵二叉树，判断该二叉树是否是平衡二叉树。
**时间：2019.10.30
**思路：略
*/

class Solution {
public:
    bool IsBalanced_Solution(TreeNode* pRoot) {
        int i;
        return IsBalanced(pRoot, &i);
    }
    
    bool IsBalanced(TreeNode* pRoot, int *depth)
    {
        if(pRoot == nullptr)
        {
            *depth = 0;
            return true;
        }
        
        int left, right;
        if(IsBalanced(pRoot->left, &left) && IsBalanced(pRoot->right, &right))
        {
            int diff = left - right;
            if(diff <= 1 && diff >= -1)
            {
                *depth = 1 + (left > right? left : right);
                return true;
            }
        }
        return false;
    }
};