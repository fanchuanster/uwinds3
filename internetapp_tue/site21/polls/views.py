from django.shortcuts import render
from django.http import HttpResponse
from .forms import SearchForm
from django.http import Http404
from .models import Student, Topic, Course
from django.shortcuts import get_object_or_404

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
    setattr(request, 'view', 'index')
    return render(request, 'polls/index.html', {'top_list': top_list})

def about(request):
    return render(request, 'polls/about.html')

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
