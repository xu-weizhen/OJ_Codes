/*
**链接：https://www.nowcoder.com/questionTerminal/b736e784e3e34731af99065031301bca
**描述：输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
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
    vector<vector<int>> result;
        
    vector<vector<int>> FindPath(TreeNode* root,int expectNumber) {
        result.clear();
        
        if(root == nullptr)
            return result;
        
        vector<int> path;
        int currentSum = 0;
        FindPath(root, expectNumber, path, currentSum);
        return result;
    }
    
    void FindPath(TreeNode* pRoot, int expectedSum, vector<int>& path, int& currentSum)
    {
        currentSum += pRoot->val;
        path.push_back(pRoot->val);
        
        bool isLeaf = pRoot->left == nullptr && pRoot->right == nullptr;
        if(currentSum == expectedSum && isLeaf)
            result.push_back(path);
        
        if(pRoot->left != nullptr)
            FindPath(pRoot->left, expectedSum, path, currentSum);
        
        if(pRoot->right != nullptr)
            FindPath(pRoot->right, expectedSum, path, currentSum);

        currentSum -= pRoot->val;
        path.pop_back();
    } 
};