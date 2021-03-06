from django.db import models

# Create your models here.
from django.db import models
from datetime import datetime
from django.contrib.auth.models import User
from django.utils import timezone

class Topic(models.Model):
    def __str__(self):
        return f"{self.name}"
    name = models.CharField(max_length=200)
    length = models.BigIntegerField(default=12)

class Course(models.Model):
    def __str__(self):
        return f"{self.title} - ${self.price}"
    def get_hours(self):
        return self.topic.length if self.hours == 0 else self.hours
    def add_hours(self, increment):
        current_hours = self.get_hours()
        self.hours = current_hours + increment
        self.save()
    title = models.CharField(max_length=200)
    topic = models.ForeignKey(Topic, related_name='courses', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    hours = models.BigIntegerField(default=0)
    for_everyone = models.BooleanField(default=True)
    description = models.TextField(blank=True)
    num_reviews = models.PositiveIntegerField(default=0)

class Student(User):
    LVL_CHOICES = [
        ('HS', 'High School'),
        ('UG', 'Undergraduate'),
        ('PG', 'Postgraduate'),
        ('ND', 'No Degree'),
    ]

    def __str__(self):
        return f"{super().__str__()}"
    def upper_case_name(self):
        """
        Student Full Name
        :return:
        """
        return self.first_name.upper() + " " + self.last_name.upper()

    def user_directory_path(self, filename):
        # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
        return 'user_{0}/{1}'.format(self.username, filename)

    photo = models.ImageField(upload_to=user_directory_path, help_text="Upload image: ", blank=True)
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
    def total_items(self):
        return len(self.courses.all())
    def total_cost(self):
        return sum([course.price for course in self.courses.all()])
    def __str__(self):
        return f"${self.total_cost()} - {self.student} - {self.order_status} - {self.order_date}"
    courses = models.ManyToManyField(Course, related_name='courses', blank=True)
    student = models.ForeignKey(Student, related_name='student', on_delete=models.CASCADE)
    order_status = models.BigIntegerField(choices=ORDER_STATUSES, default=1)
    order_date = models.DateTimeField(default=datetime.now(), blank=True)

class Review(models.Model):
    def __str__(self):
        return f"rating {self.rating} by {self.reviewer} on {self.date}"
    reviewer = models.EmailField()
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    rating = models.PositiveIntegerField()
    comments = models.TextField(blank=True)
    date = models.DateField(default=timezone.now())
    datetime = models.DateTimeField(default=timezone.datetime.now())