# Generated by Django 2.2.7 on 2020-03-30 15:18

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("home", "0003_auto_20200327_1405"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="original",
            name="_data",
        ),
        migrations.AddField(
            model_name="original",
            name="_data_one",
            field=models.BinaryField(blank=True, db_column="_data_one"),
        ),
        migrations.AddField(
            model_name="original",
            name="_data_two",
            field=models.BinaryField(blank=True, db_column="_data_two"),
        ),
    ]
