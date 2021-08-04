from datetime import datetime

from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse, HttpResponseRedirect
from .forms import SearchForm, OrderForm, ReviewForm, LoginForm
from django.http import Http404
from .models import Student, Topic, Course
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test

@login_required
def myaccount(request):
    student = Student.objects.get(username=request.user)
    return render(request, 'registration/myaccount.html', {
        'isstudent': student,
        'user': student})

def user_login(request):
    if request.user.is_authenticated:
        print('the user is authenticated')
    if request.method == 'POST':
        form = LoginForm(request.POST)
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(username=username, password=password)
        if user:
            if user.is_active:
                request.session.set_expiry(600)
                login(request, user)
                request.session['last_login'] = datetime.now()
                return HttpResponseRedirect(reverse('polls:index'))
        form.errors = True
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
        last_login = 'Your las login was more than a minute ago'
    setattr(request, 'view', 'index')
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

def student_details(request, student_id):
    try:
        student = Student.objects.get(id=student_id)
    except Student.DoesNotExist:
        raise Http404('student not found')
    return render(request, 'student_detail.html', {
        'student': student
    })

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