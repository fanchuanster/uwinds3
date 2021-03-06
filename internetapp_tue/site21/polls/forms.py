from django import forms
from django.core.validators import MinValueValidator
from polls.models import Order, Review, Student, Course
from django.core.exceptions import ValidationError

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100, required=True, label='User Name')
    password = forms.CharField(widget=forms.PasswordInput)
    errors = False

class SearchForm(forms.Form):
    LENGTH_CHOICES = [
        (8, '8 Weeks'),
        (10, '10 Weeks'),
        (12, '12 Weeks'),
        (14, '14 Weeks')
    ]
    name = forms.CharField(max_length=10, required=False, label='Student Name:')
    length = forms.TypedChoiceField(widget=forms.RadioSelect, choices=LENGTH_CHOICES, coerce=int, required=False, label='Preferred course duration:')
    max_price = forms.IntegerField(label='Maximum Price', validators=[MinValueValidator(0, message='value too small')])

class OrderForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(OrderForm, self).__init__(*args, **kwargs)
        for field in iter(self.fields):
            if field != "courses":
                self.fields[field].widget.attrs.update({
                    'class': 'form-control'
                })
    class Meta:
        model = Order
        fields = ['courses', 'student', 'order_status']
        widgets = {'courses': forms.CheckboxSelectMultiple(), 'order_type': forms.RadioSelect}
        labels = {'student': u'Student Name', 'courses': u'Available Courses'}


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['reviewer', 'course', 'rating', 'comments']
        widgets = {'course':forms.RadioSelect }
        labels = {'reviewer': u'Please end a valid email',
                  'rating':u'Rating: An integer between 1(worst) and 5(best)'}
    def clean(self):
        cleaned_data = super().clean()
        rating = cleaned_data.get("rating")
        if rating < 1 or rating > 5:
            raise ValidationError(
                "You must enter a rating between 1 and 5!"
            )

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        exclude = ['last_login', 'is_superuser', 'groups', 'user_permissions', 'is_staff', 'date_joined', 'is_active']
        labels = {
            'username':u'User Name',
            'password':u'Password',
            'first_name':u'First Name',
            'last_name':u'Last Name',
            'address':u'Address',
            'province':u'Province',
            'interested_in':u'Interesting Topics',
            'email':u'Email Address'
        }
        widgets = {
            'interested_in':forms.CheckboxSelectMultiple,
            'password': forms.PasswordInput
        }
        def get_photo_url(self):
            instance = getattr(self, 'instance', None)
            if not instance:
                return ""
            return instance.phot.url

class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = '__all__'
    def clean(self):
        cleaned_data = super().clean()
        price = cleaned_data.get("price")
        if price < 100 or price > 200:
            raise ValidationError(
                "You must enter a price between 100 and 200!"
            )
        hours = cleaned_data.get("hours")
        if hours < 10 or hours > 300:
            raise ValidationError(
                "You must enter an hours value between 30 and 300!"
            )