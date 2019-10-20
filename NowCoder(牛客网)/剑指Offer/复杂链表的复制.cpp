/*
**链接：https://www.nowcoder.com/questionTerminal/f836b2c43afc4b35ad6adc41ec941dba
**描述：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
**时间：2019.10.20
**思路：在原链表每个节点后插入一个复制出来的新节点，再拆分链表
*/

/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead)
    {
        if(pHead == NULL)
            return NULL;
        
        RandomListNode* p = pHead;
        RandomListNode* newNode;
        while(p!=NULL)
        {
            newNode = new RandomListNode(p->label);
            newNode->next = p->next;
            p->next = newNode;
            p = newNode->next;
        }
        
        p = pHead;
        while(p!=NULL)
        {
            if(p->random == NULL)
                p->next->random = NULL;
           else
                p->next->random = p->random->next;
            p = p->next->next;
        }
        
        p = pHead;
        RandomListNode* result = p->next;
        RandomListNode* resultTail = result;
        p->next = p->next->next;
        resultTail->next = NULL;
        p = p->next;
        
        while(p!=NULL)
        {
            resultTail->next = p->next;
            p->next = p->next->next;
            resultTail = resultTail->next;
            resultTail->next = NULL;
            p = p->next;
        }
        return result;
    }
};