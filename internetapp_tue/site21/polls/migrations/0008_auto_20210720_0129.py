# Generated by Django 3.2.5 on 2021-07-19 17:29

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0007_alter_order_order_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='course',
            name='num_reviews',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='order',
            name='order_date',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2021, 7, 20, 1, 29, 49, 502338)),
        ),
    ]
