/*
**链接：https://www.nowcoder.com/questionTerminal/529d3ae5a407492994ad2a246518148a
**描述：输入一个链表，输出该链表中倒数第k个结点。
**时间：2019.10.14
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
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(pListHead==nullptr || k==0)
            return nullptr;
        
        ListNode* p = pListHead;
        for(int i=0; i<k-1; i++)
        {
            if(p->next == nullptr)
                return nullptr;
            else
                p = p->next;
        }
        
        ListNode* result = pListHead;
        while(p->next!=nullptr)
        {
            p = p->next;
            result = result->next;
        }
        
        return result;
    }
};