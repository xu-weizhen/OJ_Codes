/*
**链接：https://www.nowcoder.com/questionTerminal/6e5207314b5241fb83f2329e89fdecc8
**描述：地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
**时间：2019.10.17
**思路：略
*/

class Solution1 {
public:
    struct point
    {
        int x;
        int y;
    };
    int movingCount(int threshold, int rows, int cols)
    {
        if(threshold<0 || rows<=0 || cols<=0)
            return 0;
        
        bool* visited = new bool[rows*cols];
        for(int i=0; i<rows*cols; i++)
            visited[i] =  false;
        
        
        int count = movingCountCore(threshold, rows, cols, 0, 0, visited);
        
        delete []visited;
        return count;
    }
    
    int movingCountCore(int threshold, int rows, int cols, int row, int col, bool* visited)
    {
        int count = 0;
        if(check(threshold, rows, cols, row, col, visited))
        {
            visited[row*cols+col] = true;
            
            count = 1 + movingCountCore(threshold, rows, cols, row-1, col, visited) + movingCountCore(threshold, rows, cols, row+1, col, visited) + movingCountCore(threshold, rows, cols, row, col-1, visited) + movingCountCore(threshold, rows, cols, row, col+1, visited);
        }
        
        return count;
    }
    
    bool check(int threshold, int rows, int cols, int row, int col, bool* visited)
    {
        if(row>=0 && row<rows && col>=0 && col<cols && getDigitSum(row) + getDigitSum(col) <= threshold && !visited[row*cols+col])
            return true;
        return false;
    }
    
    int getDigitSum(int number)
    {
        int sum = 0;
        while(number>0)
        {
            sum += number%10;
            number /= 10;
        }
        return sum;
    }
};


class Solution2 {
public:
    struct point
    {
        int x;
        int y;
    };
    int movingCount(int threshold, int rows, int cols)
    {
        if(threshold<0 || rows<=0 || cols<=0)
            return 0;
        
        bool* p = new bool[rows*cols];
        for(int i=0; i<rows*cols; i++)
            p[i] =  false;
        
        queue<point> q;
        point o;
        o.x = 0;
        o.y = 0;
        p[0] = true;
        q.push(o);
        point now;
        point next;
        while(!q.empty())
        {
            now = q.front();
            if(ableMove(now.x, now.y-1, threshold, rows, cols))
            {
                next.x = now.x;
                next.y = now.y-1;
                if(!p[next.x*cols+next.y])
                {
                    q.push(next);
                    p[next.x*cols+next.y] = true;
                }
            }
            if(ableMove(now.x, now.y+1, threshold, rows, cols))
            {
                next.x = now.x;
                next.y = now.y+1;
                if(!p[next.x*cols+next.y])
                {
                    q.push(next);
                    p[next.x*cols+next.y] = true;
                }
            }
            if(ableMove(now.x-1, now.y, threshold, rows, cols))
            {
                next.x = now.x-1;
                next.y = now.y;
                if(!p[next.x*cols+next.y])
                {
                    q.push(next);
                    p[next.x*cols+next.y] = true;
                }
            }
            if(ableMove(now.x+1, now.y, threshold, rows, cols))
            {
                next.x = now.x+1;
                next.y = now.y;
                if(!p[next.x*cols+next.y])
                {
                    q.push(next);
                    p[next.x*cols+next.y] = true;
                }
            }
            q.pop();
        }
        
        int result = 0;
        for(int i=0; i<rows*cols; i++)
            if(p[i])
                result ++;
			
		delete []p;
        return result;
    }
    
    bool ableMove(int row, int col, int k, int rows, int cols)
    {
        if(row<0 || col<0 || row>=rows || col>=cols)
            return false;
        return row/10+row%10+col/10+col%10>k? false:true;
    }
};