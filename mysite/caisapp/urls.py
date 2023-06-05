from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("survey/", views.survey, name="survey"),
    path("index/", views.index, name="index"),
    path("training/", views.training, name="training"),
    # path("client_to_view/", views.client_to_view, name="client_to_view"),
    path("survey/caisapp/byebye/", views.byebye, name="byebye"),
]
