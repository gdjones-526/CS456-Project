from django.urls import path
from . import views

urlpatterns = [
    # Main dashboard
    path('', views.test_dashboard, name='test_dashboard'),
    
    # Test actions
    path('run/', views.run_tests, name='run_tests'),
    path('cleanup/', views.cleanup_tests, name='cleanup_tests'),
    
    # Results
    path('results/<int:run_id>/', views.test_results, name='test_results'),
    path('status/<int:run_id>/', views.get_test_status, name='get_test_status'),
]