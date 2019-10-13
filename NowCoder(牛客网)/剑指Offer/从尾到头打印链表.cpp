/*
**链接：https://www.nowcoder.com/questionTerminal/d0267f7f55b3412ba93bd35cfa8e8035
**描述：输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
**时间：2019.10.13
**思路：略
*/

/**
*  struct ListNode {
*        int val;
*        struct ListNode *next;
*        ListNode(int x) :
*              val(x), next(NULL) {
*        }
*  };
*/
class Solution {
public:
    vector<int> printListFromTailToHead(ListNode* head) {
        stack<int> s;
        vector<int> a;
        ListNode *p = head;
        while(p!=NULL)
        {
            s.push(p->val);
            p = p->next;
        }
        while(!s.empty())
        {
            a.push_back(s.top());
            s.pop();
        }
        return a;
    }
};