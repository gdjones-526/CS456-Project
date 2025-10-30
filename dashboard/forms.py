from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from core.models import UploadedFile, AIModel, Experiment

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['file', 'description']
        widgets = {
            'description': forms.Textarea(attrs={
                'rows': 3,
                'class': 'form-control',
                'placeholder': 'Describe your dataset (optional)...'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls,.json'
            })
        }
        labels = {
            'file': 'Select Dataset File',
            'description': 'Description'
        }

    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # Validate file extension
            valid_extensions = ['.csv', '.xlsx', '.xls', '.json']
            ext = file.name[file.name.rfind('.'):].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f'Unsupported file type. Please upload CSV, Excel, or JSON files only.'
                )
            # Validate file size (max 50MB)
            if file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size cannot exceed 50MB.')
        return file


class ModelTrainingForm(forms.ModelForm):

    ALGORITHM_CHOICES = [
        ('random_forest', 'Random Forest'),
        ('logistic_regression', 'Logistic Regression'),
        ('svm', 'Support Vector Machine'),
        ('neural_network', 'Neural Network'),
        ('gradient_boosting', 'Gradient Boosting'),
    ]

    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    test_size = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05',
            'placeholder': '0.2'
        }),
        help_text='Proportion of dataset to use for testing (0.1 - 0.5)'
    )
    
    random_state = forms.IntegerField(
        initial=42,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '42'
        }),
        help_text='Random seed for reproducibility'
    )

    class Meta:
        model = AIModel
        fields = ['name', 'dataset', 'target_variable']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'My Classification Model'
            }),
            'dataset': forms.Select(attrs={'class': 'form-control'}),
            'target_variable': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'target_column_name'
            })
        }

    def __init__(self, user=None, algorithm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            # Only show datasets belonging to this user
            self.fields['dataset'].queryset = UploadedFile.objects.filter(
                user=user, 
                processed=True
            )
        if algorithm and not self.is_bound:  # <-- check that form isn't bound
            if algorithm in dict(self.ALGORITHM_CHOICES):
                self.fields['algorithm'].initial = algorithm

class ExperimentForm(forms.ModelForm):
    class Meta:
        model = Experiment
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Experiment Name'
            }),
            'description': forms.Textarea(attrs={
                'rows': 4,
                'class': 'form-control',
                'placeholder': 'Describe your experiment...'
            })
        }
        labels = {
            'name': 'Experiment Name',
            'description': 'Description'
        }