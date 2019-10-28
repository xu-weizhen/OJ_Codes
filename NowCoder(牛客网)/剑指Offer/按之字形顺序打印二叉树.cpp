/*
**链接：https://www.nowcoder.com/questionTerminal/91b69814117f4e8097390d107d2efbe0
**描述：请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
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
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        
        vector<vector<int>> result;
        vector<int> now;
        
        if(pRoot == nullptr)
            return result;

        TreeNode* p;
        stack<TreeNode*> levels[2];
        int current = 0;
        int next = 1;
        
        levels[current].push(pRoot);
        
        while(!levels[0].empty() || !levels[1].empty())
        {
            p = levels[current].top();
            levels[current].pop();
            
            now.push_back(p->val);
            
            if(current == 0)
            {
                if(p->left != nullptr)
                    levels[next].push(p->left);
                if(p->right != nullptr)
                    levels[next].push(p->right);
            }
            else
            {
                if(p->right != nullptr)
                    levels[next].push(p->right);
                if(p->left != nullptr)
                    levels[next].push(p->left);
            }
            
            if(levels[current].empty())
            {
                result.push_back(now);
                now.clear();
                current = 1 - current;
                next = 1 - next;
            }
        }
        return result;
    }
    
};