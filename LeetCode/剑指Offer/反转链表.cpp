/*
**链接：https://www.nowcoder.com/questionTerminal/75e878df47f24fdc9dc3e400ec6058ca
**描述：输入一个链表，反转链表后，输出新链表的表头。
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
    ListNode* ReverseList(ListNode* pHead) {
        ListNode* result = (ListNode*)malloc(sizeof(ListNode));
        result->next = NULL;
        ListNode* p = pHead;
        ListNode* pn;
        while(p!=NULL)
        {
            pn = p->next;
            p->next = result->next;
            result->next = p;
            p = pn;
        }
        p = result->next;
        free(result);
        return p;
    }
};