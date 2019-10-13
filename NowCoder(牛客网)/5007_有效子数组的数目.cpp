class Solution {
public:
    int validSubarrays(vector<int>& nums) {
        int result = nums.size();
	    int count = 0;
        for(int i=0; i<nums.size(); i++)
        {
            count = 0;
            for(int j=i+1; j<nums.size(); j++)
            {
                if(nums[j]>=nums[i])
                {
                    count++;
                }
                else
                    break;
            }
            result += count;
        }
       return result;
    }
};

