from django.urls import path
from . import views

urlpatterns = [
    # Authentication URLs
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Main App URLs
    path('', views.dashboard, name='dashboard'),
]