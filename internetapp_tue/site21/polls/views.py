from datetime import datetime

from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse, HttpResponseRedirect
from .forms import SearchForm, OrderForm, ReviewForm, LoginForm, StudentForm
from django.http import Http404
from .models import Student, Topic, Course, User
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings

@login_required
def myaccount(request):
    student = Student.objects.filter(username=request.user).first()
    return render(request, 'registration/myaccount.html', { 'student': student, 'media_url':settings.MEDIA_URL })

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(username=username, password=password)
        if user:
            if user.is_active:
                nextpage = request.POST.get("next")
                request.session.set_expiry(600)
                login(request, user)
                request.session['last_login'] = datetime.now()
                if nextpage:
                    print("next", nextpage)
                    return redirect(nextpage)
                else:
                    print("no next")
                    return HttpResponse("xx")
        return render(request, 'registration/login.html', {'form': form})
    else:
        return render(request, 'registration/login.html', {'form':LoginForm()})

@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect(reverse('polls:index'))

def findcourses(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            length = form.cleaned_data['length']
            max_price = form.cleaned_data['max_price']
            if length:
                topics = Topic.objects.filter(length=length)
            else:
                topics = Topic.objects.all()
            courselist = []
            for topic in topics:
                courselist = courselist + list(topic.courses.filter(price__lte=max_price))
            return render(request, 'polls/results.html', { 'courselist':courselist, 'name':name, 'length':length })
        else:
            return HttpResponse('Invalid data' + str(form.errors))
    else:
        return render(request, 'polls/findcourses.html', { 'form': SearchForm()})

def index(request):
    top_list = Topic.objects.all().order_by('id')[:10]
    last_login = request.session.get('last_login')
    if not last_login:
        last_login = 'Your last login was more than a minute ago'
    # setattr(request, 'view', 'index')
    return render(request, 'polls/index.html', {'top_list': top_list, 'last_login': "Your las login:" + str(last_login)})

def about(request):
    value = request.COOKIES.get('about_visits')
    if value is None:
        about_visits = 0
    else:
        about_visits = int(value)
    about_visits += 1
    response = render(request, 'polls/about.html', {'about_visits': about_visits})
    response.set_cookie('about_visits', about_visits, expires=5*60)
    return response

def detail(request, topic_id):
    topic = get_object_or_404(Topic, id=topic_id)
    courses = Course.objects.filter(topic__id=topic_id)
    return render(request, 'polls/detail.html',
                  {
                      'name': topic.name,
                      'length': topic.length,
                      'number_of_courses':len(courses),
                      'courses':courses})

@login_required
def place_order(request):
    if request.method == 'POST':
        form = OrderForm(request.POST)
        if form.is_valid():
            courses = form.cleaned_data['courses']
            order = form.save(commit=True)
            student = order.student
            status = order.order_status
            order.save()
            if status == 1:
                for c in order.courses.all():
                    student.registered_courses.add(c)
                student.save()
            return render(request, 'polls/order_response.html', {'courses':courses, 'order':order})
        else:
            return render(request, 'polls/place_order.html', {'form':form})
    else:
        form = OrderForm()
        return render(request, 'polls/place_order.html', {'form':form})

def review_course(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            form.save()
            course = form.cleaned_data['course']
            course.num_reviews += 1
            course.save()
            response = redirect('/myapp')
            return response
        else:
            return render(request, 'polls/reviewcourse.html', {'form': form})
    else:
        form = ReviewForm()
        return render(request, 'polls/reviewcourse.html', {'form':form})

def register(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data['username'], form.cleaned_data['password'])
            student = Student.objects.create()
            student.username = form.cleaned_data['username']
            student.first_name = form.cleaned_data['first_name']
            student.last_name = form.cleaned_data['last_name']
            student.address = form.cleaned_data['address']
            student.province = form.cleaned_data['province']
            student.email = form.cleaned_data['email']
            student.interested_in.set(form.cleaned_data['interested_in'])
            student.set_password(form.cleaned_data['password'])
            student.save()
            return render(request, 'polls/general_response.html', {'msg':"Congratulations! Registration succeeded!"})
        else:
            print("form is not valid")
            return render(request, 'polls/register.html', {'form': form})
    else:
        form = StudentForm()
        return render(request, 'polls/register.html', {'form':form})

def edit_myaccount(request):
    pass