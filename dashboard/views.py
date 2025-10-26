from django.shortcuts import render

def dashboard(request):
    context = {
        "accuracy": 94,
        "loss": 0.06
    }
    return render(request, "dashboard.html", context)