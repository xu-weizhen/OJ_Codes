class Solution {
public:
    int missingElement(vector<int>& nums, int k) {
        int count = 0;
        int t;
        t = nums[0];
        for(int i=1;i<nums.size(); i++)
        {
            while(t+1!=nums[i])
            {
                t++;
                count++;
                if(count==k)
                    return t;
            }
            t++;
        }
        return k-count+t;
    }
};

