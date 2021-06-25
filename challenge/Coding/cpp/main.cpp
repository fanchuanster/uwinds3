#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

/**
 * 1. uniform initialization with brackets.
 * 2. constructor improvement - delegation to other destructors.
 * 3. class member in-place initialization
 * 4. explicit keyword, prevent implicit converting
 * 5. range iteration with for (auto& i:vec)
 * 6. auto type inference
 * 7. for_each(first_iter, last_iter, func or Fun operator or lambda)
 * 8. lambda
 */



void myfunction(int i)
{
    cout<<i<<endl;
}

class BaseClass
{
  public:
    BaseClass(int i, const string& s):i(i), s(s)
    {
        cout<<"BaseClass("<<i<<", "<<s<<")"<<endl;
    }
    BaseClass(BaseClass&& b)
    {
        cout<<"BaseClass Move"<<endl;
        i = b.i;
        s = b.s;
    }
    explicit BaseClass(int i):BaseClass(i,"dummy")
    {
        cout<<"BaseClass("<<i<<")"<<endl;
    }
    void expr() const
    {
        cout<<"i="<<i<<", s="<<s<<endl;
    }
  protected:
    int i=1;
    string s = "ss";
};
class DerivedClass : public BaseClass
{
  public:
    DerivedClass(int i, const string& s):
    BaseClass(i)
    {
        this->i = i;
        this->s = s;
    }
};

BaseClass getClass()
{
    BaseClass b = {1,"1"};
    return b;
}

void funclass(BaseClass& b)
{
    cout<<"funclass:"<<endl;
    b.expr();
}

int main()
{
    cout<<"hello world"<<endl;

    auto bb = getClass();
    bb.expr();

    DerivedClass d = {0, "dddt"};

    BaseClass b = {2, "ttt"};
    b.expr();

    funclass(b);
    funclass(BaseClass(9));


    vector<int> v = {1,3,4,5,5,4};
    for_each(v.begin(), v.end(), [&b](int i)->int {
        b.expr();
        cout<<i*10<<endl;
        return i*10;
        });

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