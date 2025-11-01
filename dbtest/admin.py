from django.contrib import admin
from .models import TestRun, TestResult


@admin.register(TestRun)
class TestRunAdmin(admin.ModelAdmin):
    list_display = ['id', 'timestamp', 'test_type', 'status', 'passed', 'failed', 'total_tests', 'pass_rate_display', 'duration']
    list_filter = ['status', 'test_type', 'timestamp']
    search_fields = ['id']
    readonly_fields = ['timestamp', 'duration']
    date_hierarchy = 'timestamp'
    
    def pass_rate_display(self, obj):
        return f"{obj.pass_rate:.1f}%"
    pass_rate_display.short_description = 'Pass Rate'


@admin.register(TestResult)
class TestResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'test_run', 'test_case', 'test_method', 'status', 'duration_display', 'timestamp']
    list_filter = ['status', 'test_run__test_type', 'timestamp']
    search_fields = ['test_case', 'test_method', 'error_message']
    readonly_fields = ['timestamp', 'duration']

    fieldsets = (
        ('Basic Info', {
            'fields': ('test_run', 'test_case', 'test_method', 'status')
        }),
        ('Performance', {
            'fields': ('duration',)
        }),
        ('Error Details', {
            'fields': ('error_message', 'error_traceback'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('timestamp',)
        }),
    )
    
    def duration_display(self, obj):
        if obj.duration:
            return f"{obj.duration:.2f}s"
        return "-"
    duration_display.short_description = 'Duration'