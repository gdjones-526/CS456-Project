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

    # File upload and management
    path('upload/', views.upload_file, name='upload_file'),
    path('files/', lambda request: redirect('dashboard'), name='file_list'),
    path('files/<int:pk>/', views.file_detail, name='file_detail'),
    path('files/<int:pk>/delete/', views.delete_file, name='delete_file'),

    # Model training
    path('train/<int:dataset_id>/', views.train_model, name='train_model'),
    path('models/', views.model_list, name='model_list'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
]