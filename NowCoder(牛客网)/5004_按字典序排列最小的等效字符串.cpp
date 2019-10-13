class Solution {
public:
    string smallestEquivalentString(string A, string B, string S) {
        string s[26];
	for(int i=0; i<26; i++)
		s[i]="";
	
	int count = 0;
	bool find = true;

	for(int i=0; i<A.length(); i++)
	{
		find = false;
		for(int j=0; j<count && !find; j++)
		{
			for(int k=0; k<s[j].length(); k++)
			{
				if(A[i]==s[j][k])
				{
					if(B[i]<s[j][0])
						s[j].insert(s[j].begin(), B[i]);
					else
						s[j].insert(s[j].end(),B[i]);
					find = true;
					break;
				}
				if(B[i]==s[j][k])
				{
					if(A[i]<s[j][0])
						s[j].insert(s[j].begin(), A[i]);
					else
						s[j].insert(s[j].end(),A[i]);
					find = true;
					break;
				}
			}
		}

		if(!find)
		{
			if(A[i]==B[i])
				s[count].insert(s[count].begin(), A[i]);
			else if(A[i]<B[i])
			{
				s[count].insert(s[count].begin(), B[i]);
				s[count].insert(s[count].begin(), A[i]);
			}
			else
			{
				s[count].insert(s[count].begin(), A[i]);
				s[count].insert(s[count].begin(), B[i]);
			}
				

			count++;
		}
	}

	//合并
    bool ch = false;
	for(int i=0; i<count; i++)
	{
		for(int j=i+1; j<count; j++)
		{
			ch = false;
			for(int m=0; m<s[i].length()&&!ch; m++)
				for(int n=0; n<s[j].length(); n++)
				{
					if(s[i][m]==s[j][n])
					{
						if(s[i][0]>s[j][0])
						{
							s[i].insert(s[i].begin(),s[j][0]);
						}
						s[i] += s[j];
						s[j]="";
						ch = true;
						break;
					}
				}
		}
	}

	string result = "";
	for(int i=0; i<S.length(); i++)
	{
		find = false;
		for(int j=0; j<count&&!find; j++)
		{
			for(int k=0; k<s[j].length(); k++)
			{
				if(s[j][k]==S[i])
				{
					result.insert(result.end(), s[j][0]);
					find = true;
					break;
				}
			}
		}
        if(!find)
			result.insert(result.end(), S[i]);
	}

	return result;
    }
};

