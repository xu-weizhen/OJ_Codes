/*
**链接：https://www.nowcoder.com/questionTerminal/253d2c59ec3e4bc68da16833f79a38e4
**描述：给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
**时间：2019.10.17
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
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        ListNode* pfast;
        ListNode* pslow;
        
        pfast = pHead;
        pslow = pHead;
        
        if(pfast->next == NULL || pHead == NULL)
            return NULL;
        
        pfast = pfast->next->next;
        pslow = pfast->next;
        while(pfast!=NULL && pslow!=NULL && pfast!= pslow)
        {
            if(pfast->next != NULL)
                pfast = pfast->next->next;
            else
                return NULL;
            pslow = pslow->next;
        }
        
        if(pfast==NULL || pslow==NULL || pfast != pslow)
            return NULL;
        
        int count = 1;
        pfast = pfast->next;
        while(pfast != pslow)
        {
            count++;
            pfast = pfast->next;
        }
        
        pfast = pHead;
        pslow = pHead;
        for(int i=0; i<count; i++)
            pfast = pfast->next;
        
        while(pfast != pslow)
        {
            pfast = pfast->next;
            pslow = pslow->next;
        }
        
        return pslow;
    }
};