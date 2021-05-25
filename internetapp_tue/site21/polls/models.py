from django.db import models

# Create your models here.
from django.db import models
import datetime
from django.contrib.auth.models import User
from django.utils import timezone

class Topic(models.Model):
    name = models.CharField(max_length=200)

class Course(models.Model):
    title = models.CharField(max_length=200)
    topic = models.ForeignKey(Topic, related_name='courses', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    for_everyone = models.BooleanField(default=True)