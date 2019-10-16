/*
**链接：https://www.nowcoder.com/questionTerminal/564f4c26aa584921bc75623e48ca3011
**操作给定的二叉树，将其变换为源二叉树的镜像。
**时间：2019.10.14
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
    void Mirror(TreeNode *pRoot) {
        if(pRoot == nullptr)
            return;
        
        if(pRoot->left==nullptr && pRoot->right==nullptr)
            return;

        TreeNode *t = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = t;
        
        if(pRoot->left!=nullptr)
            Mirror(pRoot->left);
        
        if(pRoot->right!=nullptr)
            Mirror(pRoot->right);
    }
};