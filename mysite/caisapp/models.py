from django.db import models

# Create your models here.
from django.db import models
from django.core.exceptions import ValidationError


class Category(models.Model):
    # can be either 'video' or 'demograhpics'
    name = models.CharField(max_length=400, default="", null=True)

    def __str__(self):
        return self.name


class VideoGenerator(models.Model):
    TEXT, CAPTION = "text", "caption"
    VIDEO_TYPES = (CAPTION, "caption")

    video_name = models.CharField(
        max_length=100, blank=True, null=True, default=""
    )  # video text.
    caption_title = models.CharField(max_length=100, blank=True, null=True, default="")
    question_type = models.CharField(max_length=200, default=TEXT)

    def __str__(self):
        return self.video_name


class Question(models.Model):
    TEXT, RADIO, VIDEO = "text", "radio", "video"
    QUESTION_TYPES = ((TEXT, "text"), (RADIO, "radio"), (VIDEO, "video"))

    text = models.TextField(default="")  # question text.
    required = models.BooleanField(default=False)
    category = models.ForeignKey(
        Category, blank=True, null=True, on_delete=models.CASCADE
    )
    question_type = models.CharField(
        max_length=200, choices=QUESTION_TYPES, default=RADIO
    )
    choices = models.TextField(
        blank=True,
        null=True,
    )

    def save(self, *args, **kwargs):
        super(Question, self).save(*args, **kwargs)

    def get_choices(self):
        """parse the choices field and return a tuple formatted appropriately
        for the 'choices' argument of a form widget."""
        choices = self.choices.split(",")
        choices_list = []
        for c in choices:
            c = c.strip()
            choices_list.append((c, c))
        choices_tuple = tuple(choices_list)
        return choices_tuple

    def __str__(self):
        return self.text


class Response(models.Model):
    """
    a response object is just a collection of questions and answers with
    a unique interview uuid
    """
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    interview_uuid = models.CharField("Unique Identifier", max_length=36)


class AnswerBase(models.Model):
    category = models.ForeignKey(
        Category, blank=True, null=True, on_delete=models.CASCADE
    )
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    response = models.ForeignKey(Response, on_delete=models.CASCADE)

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)


# these type-specific answer models use a text field to allow for flexible
# field sizes depending on the actual question this answer corresponds to.
# any "required" attribute will be enforced by the form.


class AnswerText(AnswerBase):
    body = models.TextField(blank=True, null=True)


class AnswerRadio(AnswerBase):
    body = models.CharField(max_length=200, blank=True, null=True, default="")


class CaptionName(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    response = models.ForeignKey(Response, on_delete=models.CASCADE)
    caption_title = models.TextField(blank=True, null=True)


# Blobby for the QBC
class Blobby(models.Model):
    _data_one = models.BinaryField(db_column="data_one", blank=True)
    _data_two = models.BinaryField(db_column="data_two", blank=True)
    name = models.CharField(max_length=50, default="")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    response = models.ForeignKey(Response, on_delete=models.CASCADE)

    def set_data(self, filename):
        with open(filename, "rb") as file:
            self._data = file.read()

    def set_data_one(self, filename):
        with open(filename, "rb") as file:
            self._data_one = file.read()

    def set_data_two(self, filename):
        with open(filename, "rb") as file:
            self._data_two = file.read()

    def bin_to_h5(self, data):
        with open("ac_newmod.h5", "wb") as file:
            self._data = file.write(data)

    def get_data(self):
        return self._data


class AnswerVideo(AnswerBase):
    """
    Actual raw values for the caption (e.g., delay, speed, mw, pv)
    => 3000ms, 150wpm, 5 words missing, (paraphrased=1, verbatim=0)
    => 3000, 150, 5, 1

    Machine prediction values are (e.g., delay_pred)
    => 5,4,5,2

    # The user input rating in pure string type (body field)
    => 3455

    # usage
    a = AnswerVideo.objects.create(
            caption_title='caption', clip_title='v00_genre_0.vtt',
            delay=3000, speed=150, mw=5, pv=1,
            delay_pred=5, speed_pred=4, mw_pred=5, pv_pred=2,
            question=q, response=response_object, category=q.category,
            body='3455'
        )

    """

    caption_title = models.CharField(max_length=100, blank=True, null=True, default="")
    clip_title = models.CharField(max_length=100, blank=True, null=True, default="")
    delay, speed, mw, pv = (
        models.FloatField(),
        models.FloatField(),
        models.IntegerField(),
        models.IntegerField(),
    )
    delay_pred, speed_pred, mw_pred, pv_pred = (
        models.FloatField(),
        models.FloatField(),
        models.FloatField(),
        models.IntegerField(),
    )
    body = models.CharField(max_length=200, blank=True, null=True, default="")

    def set_values(
        self, caption_title, clip_title, raw_vals, machine_pred, user_ratings
    ):
        self.delay, self.speed, self.mw, self.pv = (
            raw_vals[0],
            raw_vals[1],
            raw_vals[2],
            raw_vals[3],
        )
        self.delay_pred, self.speed_pred, self.mw_pred, self.pv_pred = (
            machine_pred[0],
            machine_pred[1],
            machine_pred[2],
            machine_pred[3],
        )
        self.body = user_ratings
