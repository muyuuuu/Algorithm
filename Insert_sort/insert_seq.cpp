/*** 
 * @Author         : lanling
 * @Date           : 2020-04-13 14:23:01
 * @LastEditTime   : 2020-04-13 14:29:07
 * @FilePath       : \Insert_sort\insert_seq.cpp
 * @Github         : https://github.com/muyuuuu
 * @Description    : 顺序结构计时
 * @佛祖保佑，永无BUG
 */

#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

int arr[1000002] = {0};

int main(int argc, char const *argv[])
{
    int len = 0;

    // path 是读取的文件
    string path = "Three_hundred_thousand.txt";
    // out 是输出结构的文件
    string out = "Three_hundred_thousand_seq.txt";
    
    ifstream in(path.c_str());
    ofstream ou(out.c_str());
    
    if (!in.is_open())
    {
        cerr << "open file failed!" << endl;
        exit(-1);
    }

    if (!ou.is_open())
    {
        cerr << "create file failed!" << endl;
        exit(-1);
    }
    
    // 读取文件 存入数组
    int a = 0;
    while (in >> a)
        arr[len++] = a;
    
    int temp;
    // 开始计时
    clock_t start = clock();
    for (int i = 1; i < len; i++)
    {
        if (arr[i] >= arr[i - 1]){
            continue;
        }
        else{
            temp = arr[i];
            int k = i;
            for (int j = i - 1; j >= 0; j--){
                if (arr[j] <= arr[i] || (j == 0 && arr[j] > arr[i])){   
                    if (j == 0 && arr[j] > arr[i]) 
                        j = -1;
                    while (k > j + 1){
                        arr[k] = arr[k - 1];
                        k--;
                    }
                    arr[j + 1] = temp;
                    break;
                }
            }
        }
    }
    clock_t end   = clock();
    cout <<  (double)(end - start) / CLOCKS_PER_SEC << " seconds" << endl;
    in.close();
    // 把结果记录到文件 检查算法是否正确
    for (int i = 0; i < len; i++)
        ou << arr[i] << endl;
    ou.close();
    return 0;
}