/*
**链接：https://www.nowcoder.com/questionTerminal/8a19cbe657394eeaac2f6ea9b0f6fcf6
**描述：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
**时间：2019.10.22
**思路：略
*/

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        TreeNode* r = build(pre, vin);
        return r;
    }
    
    TreeNode* build(vector<int> pre,vector<int> vin)
    {
        if(pre.empty())
            return nullptr;
        
        TreeNode* r = new TreeNode(pre[0]);
        
        if(pre.size() == 1)
        {
            r->left = nullptr;
            r->right = nullptr;
            return r;
        }
        
        vector<int> leftPre;
        vector<int> leftVin;
        vector<int> rightPre;
        vector<int> rightVin;
        
        int pnode = 0;
        int count = 0;
        while(vin[pnode] != pre[0])
        {
            leftVin.push_back(vin[pnode]);
            count++;
            pnode++;
        }
        
        for(int i=pnode+1; i<vin.size(); i++)
            rightVin.push_back(vin[i]);
        
        for(int i=count+1; i<pre.size(); i++)
            rightPre.push_back(pre[i]);
        
        for(int i=1; count>0; i++,count--)
            leftPre.push_back(pre[i]);
        
        if(leftPre.empty())
            r->left = nullptr;
        else
            r->left = build(leftPre, leftVin);
        
        if(rightPre.empty())
            r->right = nullptr;
        else
            r->right = build(rightPre, rightVin);
        
        return r;
    }
};