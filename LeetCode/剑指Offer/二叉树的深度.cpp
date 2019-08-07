/*
**链接：https://www.nowcoder.com/questionTerminal/435fb86331474282a3499955f0a41e8b
**描述：输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
**时间：2019.8.7
**思路：略
*/

/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/

class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot == NULL)
            return 0;
        int leftdeep = TreeDepth(pRoot->left);
        int rightdeep = TreeDepth(pRoot->right);
        if(leftdeep > rightdeep)
            return leftdeep+1;
        else
            return rightdeep+1;
    }
};