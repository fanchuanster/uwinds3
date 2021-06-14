from django.shortcuts import render
from django.http import HttpResponse
from django.http import Http404
from .models import Student

def home(request):
    return HttpResponse('<p>home view</>')

def student_details(request, student_id):
    try:
        student = Student.objects.get(user_ptr_id=student_id)
    except Student.DoesNotExist:
        raise Http404('student not found')
    return render(request, 'student_detail.html', {
        'student': student
    })
