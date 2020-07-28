from django import forms
from django.forms import ModelForm
from .models import Response, Question, Category, AnswerRadio, AnswerText
import uuid
from django.http import HttpResponseRedirect

# forms.ModelForm, form is being used directly add or edit a Django model, can use ModelForm to avoid duplicating
# model description
class ResponseForm(forms.ModelForm):
    class Meta:
        model = Response
        # exclude = ['question_type', 'category', 'required', "choices", "text"]
        exclude = ["created", "interview_uuid"]

    def __init__(self, *args, **kwargs):
        # expects a survey object to be passed in initially
        self.uuid = random_uuid = uuid.uuid4().hex  # generate unique uuid
        super(ResponseForm, self).__init__(*args)

        # Represent model classes as widgets
        question = Question.objects.all()
        # print(question)
        for q in question:
            if q.question_type == Question.RADIO:
                self.fields["question_{}".format(q.pk)] = forms.TypedChoiceField(
                    label=q.text,
                    choices=q.get_choices(),
                    widget=forms.RadioSelect(attrs={}),
                )
            elif q.question_type == Question.TEXT:
                self.fields["question_{}".format(q.pk)] = forms.CharField(
                    label=q.text,
                    max_length=500,
                    initial=".",
                    widget=forms.Textarea(attrs={"class": "commentbox"}),
                )
            else:
                pass  # Prevents split error for name="videoresp" of Question table.

    def save(self, commit=True):
        # save the response object
        response = super(ResponseForm, self).save(commit=False)
        response.interview_uuid = self.uuid
        response.save()

        for field_name, field_value in self.cleaned_data.items():
            if field_name.startswith("question_"):
                # warning: this way of extracting the id is very fragile and
                # entirely dependent on the way the question_id is encoded in the
                # field name in the __init__ method of this form class.
                q_id = int(field_name.split("_")[1])
                q = Question.objects.get(pk=q_id)

                if q.question_type == Question.RADIO:
                    a = AnswerRadio(question=q)
                    a.body = field_value
                    a.category = q.category
                elif q.question_type == Question.TEXT:
                    a = AnswerText(question=q)
                    a.body = field_value

                a.question = q
                a.response = response
                a.save()
        return response
