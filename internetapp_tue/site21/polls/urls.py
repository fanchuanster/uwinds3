"""site21 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from polls import views
from django.contrib.auth import views as auth_views

app_name="polls"
urlpatterns = [
    path(r'', views.index, name='index'),
    # the path should not prefixed with /
    # path('student/<int:student_id>/', views.student_details, name='student_details')
    path('about/', views.about, name='about1'),
    path('<int:topic_id>/', views.detail, name='detail'),
    path('findcourses', views.findcourses, name='findcourses'),
    path('place_order', views.place_order, name='placeorder'),
    path('review', views.review_course, name='reviewcourse'),
    # path('user_login', views.user_login, name='user_login'),
    # url('login/','django.contrib.auth.views.login', {'template_name': 'registration/login.html'}),
    path('user_login/', auth_views.LoginView.as_view(), name='user_login'),
    path('user_logout', views.user_logout, name='user_logout'),
    path('myaccount/', views.myaccount, name='myaccount'),
    path('register/', views.register, name='register'),
]
