from django.contrib import admin

from .models import UploadedFile, AIModel, PerformanceMetric, Prediction, Figure, Experiment

admin.site.register(UploadedFile)
admin.site.register(AIModel)
admin.site.register(PerformanceMetric)
admin.site.register(Prediction)
admin.site.register(Figure)
admin.site.register(Experiment)
