/*
**链接：https://www.nowcoder.com/questionTerminal/6ab1d9a29e88450685099d45c9e31e46
**描述：输入两个链表，找出它们的第一个公共结点。
**时间：2019.10.24
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
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        int length1 = 0;
        int length2 = 0;
        ListNode* p1 = pHead1;
        ListNode* p2 = pHead2;
        
        while(p1 != nullptr)
        {
            length1 += 1;
            p1 = p1->next;
        }
        
        while(p2 != nullptr)
        {
            length2 += 1;
            p2 = p2->next;
        }
        
        p1 = pHead1;
        p2 = pHead2;
        if(length1 > length2)
            for(int i=0; i<length1 -  length2; i++)
                p1 = p1->next;
        else
            for(int i=0; i<length2 -  length1; i++)
                p2 = p2->next;
        
        while(p1 != p2)
        {
            p1 = p1->next;
            p2 = p2->next;
        }
        
        return p1;
    }
};