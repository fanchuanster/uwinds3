from django.db import models

# Create your models here.
from django.db import models
from datetime import datetime
from django.contrib.auth.models import User
from django.utils import timezone

class Topic(models.Model):
    def __str__(self):
        return f"{self.name}, {self.length}"
    name = models.CharField(max_length=200)
    length = models.BigIntegerField(default=12)

class Course(models.Model):
    def __str__(self):
        return f"{self.title}, {self.topic}, {self.price}"
    title = models.CharField(max_length=200)
    topic = models.ForeignKey(Topic, related_name='courses', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    for_everyone = models.BooleanField(default=True)
    description = models.TextField(blank=True)

class Student(User):
    LVL_CHOICES = [
        ('HS', 'High School'),
        ('UG', 'Undergraduate'),
        ('PG', 'Postgraduate'),
        ('ND', 'No Degree'),
    ]

    def __str__(self):
        return f"student {super().__str__()}"
    level = models.CharField(choices=LVL_CHOICES, max_length=2, default='HS')
    address = models.CharField(max_length=300, blank=True)
    province=models.CharField(max_length=2, default='ON')
    registered_courses = models.ManyToManyField(Course, blank=True)
    interested_in = models.ManyToManyField(Topic)

class Order(models.Model):
    ORDER_STATUSES = [
        (0, 'Cancelled'),
        (1, 'Confirmed'),
        (2, 'On Hold')
    ]
    def total_cost(self):
        return sum([course.price for course in self.courses.all()])
    def __str__(self):
        return f"${self.total_cost()} - {self.student} - {self.order_status} - {self.order_date}"
    courses = models.ManyToManyField(Course, related_name='courses', blank=True)
    student = models.ForeignKey(Student, related_name='student', on_delete=models.CASCADE)
    order_status = models.BigIntegerField(choices=ORDER_STATUSES, default=1)
    order_date = models.DateTimeField(default=datetime.now(), blank=True)