from django.contrib import admin
from .models import (
    Question,
    Category,
    Response,
    AnswerText,
    AnswerRadio,
    AnswerVideo,
    Blobby,
    VideoGenerator,
    CaptionName,
)

from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

# Register your models here.


class CategoryAdmin(admin.ModelAdmin):
    model = Category
    extra = 0


class QuestionAdmin(admin.ModelAdmin):
    model = Question
    ordering = ("category",)
    extra = 0


class VideoGeneratorAdmin(admin.ModelAdmin):
    model = VideoGenerator


class AnswerBaseInline(admin.StackedInline):
    fields = ("question", "body", "category")
    # readonly_fields = ('question', 'body', 'category')
    extra = 0


class AnswerTextInline(AnswerBaseInline):
    model = AnswerText


class AnswerRadioInline(AnswerBaseInline):
    model = AnswerRadio


class AnswerVideoInline(admin.StackedInline):
    model = AnswerVideo
    extra = 0


class BlobbyInline(admin.StackedInline):
    model = Blobby
    list_display = ("created", "_data_one", "_data_two")
    readonly_fields = (
        "_data_one",
        "_data_two",
    )
    extra = 0


class CaptionNameInline(admin.StackedInline):
    model = CaptionName
    list_display = "created"
    readonly_fields = ("caption_title",)
    extra = 0


class ResponseAdmin(admin.ModelAdmin):
    list_display = ("interview_uuid", "created")
    inlines = [
        AnswerTextInline,
        AnswerRadioInline,
        AnswerVideoInline,
        BlobbyInline,
        CaptionNameInline,
    ]
    # specifies the order as well as which fields to act on
    # readonly_fields = ('interview_uuid', 'created', 'updated')


admin.site.register(Category, CategoryAdmin)
admin.site.register(Question, QuestionAdmin)
admin.site.register(Response, ResponseAdmin)
admin.site.register(VideoGenerator, VideoGeneratorAdmin)
