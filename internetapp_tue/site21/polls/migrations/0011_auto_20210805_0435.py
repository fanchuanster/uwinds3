# Generated by Django 3.2.5 on 2021-08-04 20:35

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0010_auto_20210722_0416'),
    ]

    operations = [
        migrations.AddField(
            model_name='course',
            name='hours',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='order',
            name='order_date',
            field=models.DateTimeField(blank=True, default=datetime.datetime(2021, 8, 5, 4, 35, 32, 577488)),
        ),
        migrations.AlterField(
            model_name='review',
            name='date',
            field=models.DateField(default=datetime.datetime(2021, 8, 4, 20, 35, 32, 578519, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='review',
            name='datetime',
            field=models.DateTimeField(default=datetime.datetime(2021, 8, 5, 4, 35, 32, 578519)),
        ),
    ]