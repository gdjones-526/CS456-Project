from django.shortcuts import render

def dashboard(request):
    models = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "Neural Network",
    ]
    context = {
        "models": models,
    }
    return render(request, "dashboard.html", context)