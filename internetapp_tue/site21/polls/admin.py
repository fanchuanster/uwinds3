from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Topic, Course
# Register your models here. admin.site.register(Topic)
admin.site.register(Course)
admin.site.register(Topic)