/*
**链接：https://www.nowcoder.com/questionTerminal/fc533c45b73a41b0b44ccba763f866ef
**描述：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
**时间：2019.10.22
**思路：略
*/

/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead == NULL || pHead->next == NULL)
            return pHead;
        
        int val;
        
        while(pHead != nullptr && pHead->next != nullptr && pHead->val == pHead->next->val)
        {
            val = pHead->val;
            while(pHead != nullptr && pHead->val == val)
                pHead = pHead->next;
        }
        
        if(pHead == NULL || pHead->next == NULL)
            return pHead;
        
        ListNode* now = pHead;
        ListNode* r;
        while(now->next != nullptr)
        {
            if(now->next->next != nullptr && now->next->val == now->next->next->val)
            {
                r = now->next;
                val = r->val;
                while(r != nullptr && r->val == val)
                    r = r->next;
                now->next = r;
            }
            else
                now = now->next;
        }
        return pHead;
    }
};