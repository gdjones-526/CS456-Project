from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class LinearRegression(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store LinearRegression_Calc

    # def LinearRegression_Calc(self):
        # add a method to calculate Linear Regression using scikit learn

    # def display_LinearRegression(self):
        # display the model graphs or data from data member

class LogisticRegression(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store LogisticRegression_Calc

    # def LogisticRegression_Calc(self):
        # add a method to calculate Logistic Regression using scikit learn

    # def display_LogisticRegression(self):
        # display the model graphs or data from data member

class DecisionTree(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store DecisionTree_Calc

    # def DecisionTree_Calc(self):
        # add a method to calculate Decision Tree using scikit learn

    # def display_DecisionTree(self):
        # display the model graphs or data from data member

class RandomForest(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store RandomForest_Calc

    # def RandomForest_Calc(self):
        # add a method to calculate Random Forest using scikit learn

    # def display_RandomForest(self):
        # display the model graphs or data from data member

class GradientBoosting(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store GradientBoosting_Calc

    # def GradientBoosting_Calc(self):
        # add a method to calculate Gradient Boosting using scikit learn

    # def display_GradientBoosting(self):
        # display the model graphs or data from data member

class SVM(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store SVM_Calc

    # def SVM_Calc(self):
        # add a method to calculate SVM using scikit learn

    # def display_SVM(self):
        # display the model graphs or data from data member

class NeuralNetwork(models.Model):
    #pull user and data from database
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    #need to reference data here with data = models.something
    #add data member to store NeuralNetwork_Calc

    # def NeuralNetwork_Calc(self):
        # add a method to calculate Neural Network using pytorch

    # def display_NeuralNetwork(self):
        # display the model graphs or data from data member