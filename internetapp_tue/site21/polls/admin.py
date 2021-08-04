from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Topic, Course, Student, Order, Review
# admin.site.register(Topic)
# admin.site.register(Student)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    fields = ['courses', ('student', 'order_status', 'order_date')]
    list_display = ['id', 'student', 'order_status', 'order_date', 'total_items', 'total_cost']

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    fields = [('title', 'topic'), ('price', 'num_reviews')]
    list_display = ('title', 'topic', 'price')

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    fields = ['reviewer', 'course', 'rating', 'comments', 'date']
    list_display = ('reviewer', 'course', 'rating', 'comments', 'date')

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    fields = ['name', 'length']
    list_display = ('name', 'length')

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    fields = ['level', 'address', 'province', 'registered_courses', 'interested_in']
    list_display = ('username', 'level', 'address', 'province')