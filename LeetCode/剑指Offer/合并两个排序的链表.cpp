/*
**链接：https://www.nowcoder.com/questionTerminal/d8b6b4358f774294a89de2a6ac4d9337
**描述：输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
**时间：2019.8.8
**思路：略
*/

/*
struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};*/
class Solution {
public:
    ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
    {
        ListNode* p1;
        ListNode *p2, *p2n;
        if(pHead1->val <= pHead1->val)
        {
            p1 = pHead1;
            p2 = pHead2;
            pHead1 = p1;
        }
        else
         {
            p2 = pHead1;
            p1 = pHead2;
            pHead1 = p2;
        }
        
        while(p2!=NULL && p1->next!=NULL)
        {
            while(p1->next!=NULL)
            {
                if(p1->val<=p2->val && p1->next->val>p2->val)
                {
                    p2n = p2->next;
                    p2->next = p1->next;
                    p1->next = p2;
                    p2 = p2n;
                    break;
                }
                p1 = p1->next;
            }
        }
        
        if(p2!=NULL && p1->next==NULL)
        {
            p1->next = p2;
        }
        
        return pHead1;
    }
};