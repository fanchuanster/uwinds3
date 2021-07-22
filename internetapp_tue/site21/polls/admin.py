from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Topic, Course, Student, Order, Review
# Register your models here. admin.site.register(Topic)
# admin.site.register(Course)
admin.site.register(Topic)
admin.site.register(Student)
admin.site.register(Review)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    fields = ['courses', ('student', 'order_status', 'order_date')]
    list_display = ['id', 'student', 'order_status', 'order_date', 'total_items']

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    fields = [('title', 'topic'), ('price', 'num_reviews')]
    list_display = ('title', 'topic', 'price')
    # change_list_template =