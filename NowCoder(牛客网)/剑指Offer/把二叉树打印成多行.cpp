/*
**链接：https://www.nowcoder.com/questionTerminal/445c44d982d04483b04a54f298796288
**描述：从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
**时间：2019.10.24
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
            
            if(pRoot == nullptr)
                return result;
            
            queue<TreeNode *> que;
            que.push(pRoot);
            
            TreeNode *p;
            
            vector<int> vec;
            int count_now = 1;
            int count_next = 0;
            
            while(!que.empty())
            {
                p = que.front();
                count_now--;
                vec.push_back(p->val);
                
                if(p->left != nullptr)
                {
                    que.push(p->left);
                    count_next++;
                }
                
                if(p->right != nullptr)
                {
                    que.push(p->right);
                    count_next++;
                }
                
                if(count_now == 0)
                {
                    count_now = count_next;
                    count_next = 0;
                    result.push_back(vec);
                    vec.clear();
                }
                
                que.pop();
            }
            return result;
        }
};