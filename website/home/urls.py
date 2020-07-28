from django.urls import path
from . import views



urlpatterns = [
    # path('', views.index, name='index'),
    path('client_to_view/', views.client_to_view, name='client_to_view'),
    # path('client_to_view/byebye', views.byebye, name='client_to_view'),

    path('', views.consent, name='consent'),
    path('survey/', views.question_view, name='question_view'),
    path('survey/home/', views.index, name='index'),
    path('survey/home/byebye/', views.byebye, name='byebye'),
    path('noconsent/', views.noconsent, name='noconsent'),
]