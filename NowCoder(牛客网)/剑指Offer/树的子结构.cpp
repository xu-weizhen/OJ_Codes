/*
**链接：https://www.nowcoder.com/questionTerminal/6e196c44c7004d15b1610b9afca8bd88
**描述：输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
**时间：2019.10.18
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
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1 == NULL || pRoot2 == NULL)
           return false;
        
        return FindSubtree(pRoot1, pRoot2);
    }
    
    bool FindSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(IsSubtree(pRoot1, pRoot2))
            return true;
        else if(pRoot1->left != NULL && FindSubtree(pRoot1->left, pRoot2))
            return true;
        else if(pRoot1->right != NULL && FindSubtree(pRoot1->right, pRoot2))
            return true;
        return false;
    }
    
    bool IsSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        if(pRoot1 == NULL && pRoot2 == NULL)
            return true;
        if((pRoot1 == NULL && pRoot2 != NULL) || (pRoot1 != NULL && pRoot2 == NULL))
            return false;
        if(pRoot1->val != pRoot2->val)
            return false;

        bool left = IsSubtree(pRoot1->left, pRoot2->left) || pRoot2->left == NULL;
        bool right = IsSubtree(pRoot1->right, pRoot2->right) || pRoot2->right == NULL;
        return left && right;
    }
};