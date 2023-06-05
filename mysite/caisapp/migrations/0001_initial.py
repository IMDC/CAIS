# Generated by Django 4.2.1 on 2023-05-29 17:56

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AnswerBase",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="Category",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(default="", max_length=400, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Response",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True)),
                (
                    "interview_uuid",
                    models.CharField(max_length=36, verbose_name="Unique Identifier"),
                ),
            ],
        ),
        migrations.CreateModel(
            name="VideoGenerator",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "video_name",
                    models.CharField(blank=True, default="", max_length=100, null=True),
                ),
                (
                    "caption_title",
                    models.CharField(blank=True, default="", max_length=100, null=True),
                ),
                ("question_type", models.CharField(default="text", max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name="AnswerRadio",
            fields=[
                (
                    "answerbase_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="caisapp.answerbase",
                    ),
                ),
                (
                    "body",
                    models.CharField(blank=True, default="", max_length=200, null=True),
                ),
            ],
            bases=("caisapp.answerbase",),
        ),
        migrations.CreateModel(
            name="AnswerText",
            fields=[
                (
                    "answerbase_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="caisapp.answerbase",
                    ),
                ),
                ("body", models.TextField(blank=True, null=True)),
            ],
            bases=("caisapp.answerbase",),
        ),
        migrations.CreateModel(
            name="AnswerVideo",
            fields=[
                (
                    "answerbase_ptr",
                    models.OneToOneField(
                        auto_created=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        parent_link=True,
                        primary_key=True,
                        serialize=False,
                        to="caisapp.answerbase",
                    ),
                ),
                (
                    "caption_title",
                    models.CharField(blank=True, default="", max_length=100, null=True),
                ),
                (
                    "clip_title",
                    models.CharField(blank=True, default="", max_length=100, null=True),
                ),
                ("delay", models.FloatField()),
                ("speed", models.FloatField()),
                ("mw", models.IntegerField()),
                ("pv", models.IntegerField()),
                ("delay_pred", models.FloatField()),
                ("speed_pred", models.FloatField()),
                ("mw_pred", models.FloatField()),
                ("pv_pred", models.IntegerField()),
                (
                    "body",
                    models.CharField(blank=True, default="", max_length=200, null=True),
                ),
            ],
            bases=("caisapp.answerbase",),
        ),
        migrations.CreateModel(
            name="Question",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("text", models.TextField(default="")),
                ("required", models.BooleanField(default=False)),
                (
                    "question_type",
                    models.CharField(
                        choices=[
                            ("text", "text"),
                            ("radio", "radio"),
                            ("video", "video"),
                        ],
                        default="radio",
                        max_length=200,
                    ),
                ),
                ("choices", models.TextField(blank=True, null=True)),
                (
                    "category",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="caisapp.category",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="CaptionName",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True)),
                ("caption_title", models.TextField(blank=True, null=True)),
                (
                    "response",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="caisapp.response",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Blobby",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("_data_one", models.BinaryField(blank=True, db_column="data_one")),
                ("_data_two", models.BinaryField(blank=True, db_column="data_two")),
                ("name", models.CharField(default="", max_length=50)),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("updated", models.DateTimeField(auto_now=True)),
                (
                    "response",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="caisapp.response",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="answerbase",
            name="category",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="caisapp.category",
            ),
        ),
        migrations.AddField(
            model_name="answerbase",
            name="question",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="caisapp.question"
            ),
        ),
        migrations.AddField(
            model_name="answerbase",
            name="response",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE, to="caisapp.response"
            ),
        ),
    ]
