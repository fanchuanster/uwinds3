from django import forms
from django.core.validators import MinValueValidator
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