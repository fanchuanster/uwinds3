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

@admin.action(description='Add 50 hours')
def add_50_to_hours(courseadmin, request, queryset):
    for obj in queryset:
        obj.add_hours(50)

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    fields = [('title', 'topic'), ('price', 'num_reviews')]
    list_display = ('title', 'topic', 'price', 'chours', 'for_everyone')
    actions = [add_50_to_hours]
    def chours(self, obj):
        return obj.get_hours()

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