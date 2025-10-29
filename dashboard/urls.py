from django.urls import path
from django.shortcuts import redirect
from . import views

urlpatterns = [
    # Authentication URLs
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Main App URLs
    path('', views.dashboard, name='dashboard'),

    path('upload/', views.upload_file, name='upload_file'),
    path('files/', lambda request: redirect('dashboard'), name='file_list'),
    path('files/<int:pk>/', views.file_detail, name='file_detail'),
    path('files/<int:pk>/delete/', views.delete_file, name='delete_file'),
]