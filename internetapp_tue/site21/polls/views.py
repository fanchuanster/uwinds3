from django.shortcuts import render
from django.http import HttpResponse
from django.http import Http404
from .models import Student, Topic, Course
from django.shortcuts import get_object_or_404
#
# def index(request):
#     topics = Topic.objects.all().order_by('id')[:10]
#     response = HttpResponse()
#     heading = '<p>' + 'List of topics:' + '</p>'
#     response.write(heading)
#     for topic in topics:
#         topic_str = '<p>' + str(topic.id) + ': ' + str(topic) + '<p>'
#         response.write(topic_str)
#     courses = Course.objects.all().order_by('-title')[:5]
#     heading = '<p>' + 'List of courses:' + '</p>'
#     response.write(heading)
#     for course in courses:
#         course_str = '<p>' + str(course.title) + ': ' + str(course.price) + '<p>'
#         response.write(course_str)
#     return response


def index(request):
    top_list = Topic.objects.all().order_by('id')[:10]
    return render(request, 'polls/index.html', {'top_list': top_list})

def about(request):
    # return HttpResponse('This is an E-learning Website! Search our Topics to find all available Courses.')
    return render(request, 'polls/about0.html')

def detail(request, topic_id):
    topic = get_object_or_404(Topic, id=topic_id)
    courses = Course.objects.filter(topic__id=topic_id)
    return render(request, 'polls/detail0.html',
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
