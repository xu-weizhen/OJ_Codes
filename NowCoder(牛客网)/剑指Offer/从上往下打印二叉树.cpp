/*
**链接：https://www.nowcoder.com/questionTerminal/7fe2212963db4790b57431d9ed259701
**描述：从上往下打印出二叉树的每个节点，同层节点从左至右打印。
**时间：2019.10.22
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
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        queue<TreeNode*> que;
        vector<int> result;
        
        if(root == nullptr)
            return result;
        
        que.push(root);
        TreeNode* t;
        
        while(!que.empty())
        {
            t = que.front();
            result.push_back(t->val);
            if(t->left != nullptr)
                que.push(t->left);
            if(t->right != nullptr)
                que.push(t->right);
            que.pop();
        }
        
        return result;
    }
};