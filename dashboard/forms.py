from django import forms
from core.models import UploadedFile, AIModel, Experiment
from .ml_utils import ModelRegistry

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
                'accept': '.csv,.xlsx,.xls,.json,.txt'
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
            valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.txt']
            ext = file.name[file.name.rfind('.'):].lower()
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f'Unsupported file type. Please upload CSV, Excel, JSON, or TXT files only.'
                )
            # Validate file size (max 50MB)
            if file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size cannot exceed 50MB.')
        return file


class ModelTrainingForm(forms.ModelForm):
    # Build algorithm choices from registry (classification + regression)
    ALGORITHM_CHOICES = []
    # merge classification and regression models; use human-friendly name if available
    combined = {**ModelRegistry.CLASSIFICATION_MODELS, **ModelRegistry.REGRESSION_MODELS}
    for key, meta in combined.items():
        display = meta.get('name', key.replace('_', ' ').title())
        ALGORITHM_CHOICES.append((key, display))

    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    task_type = forms.ChoiceField(
        choices=[
            ('classification', 'Classification'),
            ('regression', 'Regression'),
        ],
        widget=forms.HiddenInput(),
        required=False  # Make it optional so we can fallback to auto-detection
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
    
    validation_size = forms.FloatField(
        initial=0.0,
        min_value=0.0,
        max_value=0.3,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05',
            'placeholder': '0.0'
        }),
        help_text='Proportion for validation set (0.0 - 0.3, optional)'
    )
    
    missing_value_strategy = forms.ChoiceField(
        choices=[
            ('mean', 'Mean Imputation'),
            ('median', 'Median Imputation'),
            ('mode', 'Mode Imputation'),
            ('drop', 'Drop Rows with Missing Values'),
        ],
        initial='mean',
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Strategy for handling missing values'
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

    def __init__(self, user=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            # Only show datasets belonging to this user
            self.fields['dataset'].queryset = UploadedFile.objects.filter(
                user=user, 
                processed=True
            )


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