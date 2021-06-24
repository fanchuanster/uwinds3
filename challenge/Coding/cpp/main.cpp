#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

void myfunction(int i)
{
    cout<<i<<endl;
}

class BaseClass
{
  public:
    BaseClass(int i, const string& s):i(i), s(s)
    {}
    BaseClass(int i):BaseClass(i,"dummy")
    {
    }
    void expr()
    {
        cout<<"i="<<i<<", s="<<s<<endl;
    }
  private:
    int i=1;
    string s = "ss";
};

int main()
{
    cout<<"hello world"<<endl;

    BaseClass b = {2};
    b.expr();


    vector<int> v = {1,3,4,5,5,4};
    for_each(v.begin(), v.end(), myfunction);

    for (auto& i:v)
    {
        cout<<"for "<<i<<endl;
    }

    string s = "s1";
    string s2("s2");
    cout<<s<<",s2"<<endl;

    int ints[] = {1,2,3,4};
    for (auto& i:ints)
    {
        i *= 2;
    }
    for (auto& i:ints)
    {
        cout<<i<<endl;
    }
}