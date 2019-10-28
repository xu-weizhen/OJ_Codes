/*
**链接：https://www.nowcoder.com/questionTerminal/947f6eb80d944a84850b0538bf0ec3a5
**描述：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
**时间：2019.10.28
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
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        TreeNode *pLastNodeInList = nullptr;
        ConvertNode(pRootOfTree, &pLastNodeInList);
        
        TreeNode *pHeadOfList = pLastNodeInList;
        while(pHeadOfList != nullptr && pHeadOfList->left != nullptr)
            pHeadOfList = pHeadOfList->left;
        
        return pHeadOfList;
    }
    
    void ConvertNode(TreeNode* pNode, TreeNode** pLastNodeInList)
    {
        if(pNode == nullptr)
            return;

        TreeNode *pCurrent = pNode;

        if (pCurrent->left != nullptr)
            ConvertNode(pCurrent->left, pLastNodeInList);

        pCurrent->left = *pLastNodeInList; 
        if(*pLastNodeInList != nullptr)
            (*pLastNodeInList)->right = pCurrent;

        *pLastNodeInList = pCurrent;

        if (pCurrent->right != nullptr)
            ConvertNode(pCurrent->right, pLastNodeInList);
    }
    
};