/*
**链接：https://www.nowcoder.com/questionTerminal/9023a0c988684a53960365b889ceaf5e
**描述：给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
**时间：2019.10.27
**思路：略
*/

/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if(pNode == nullptr)
            return nullptr;
        
        TreeLinkNode* pNext = nullptr;
        
        if(pNode->right != nullptr)
        {
            TreeLinkNode* pRight = pNode->right;
            while(pRight->left != nullptr)
                pRight = pRight->left;
            
            pNext = pRight;
        }
        else if(pNode->next != nullptr)
        {
            TreeLinkNode* pCurrent = pNode;
            TreeLinkNode* pParent = pNode->next;
            while(pParent != nullptr && pCurrent == pParent->right)
            {
                pCurrent = pParent;
                pParent = pParent->next;
            }
            
            pNext = pParent;
        }
        
        return pNext;
    }
};