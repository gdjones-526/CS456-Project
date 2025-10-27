from django.urls import path
from . import views

urlpatterns = [
    path('', views.database_test, name='dbtest'),
]