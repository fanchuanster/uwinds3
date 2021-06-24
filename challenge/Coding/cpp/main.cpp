#include <iostream>
#include <vector>

using namespace std;

void myfunction(int i)
{
    cout<<i<<endl;
}

int main()
{
    cout<<"hello world"<<endl;

    vector<int> v = {1,3,4,5,5,4};
    for_each(v.begin(), v.end(), myfunction);

    for (auto& i:v)
    {
        cout<<"for "<<i<<endl;
    }
}