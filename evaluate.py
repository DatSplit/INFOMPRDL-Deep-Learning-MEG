from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Evaluator():
    def __init__(self, predictions, targets, class_labels = ['R','WM','M','SM']):
        self.predictions = predictions
        self.targets = targets
        self.class_labels = class_labels # default is: rest,working memory,motor, story math

    def ModelSummary(self):
        "Prints all summary statistics for quick inspection"        

        self.ConfusionMatrix()
        print(f"Model accuracy is: {self.Accuracy()}")
        print(f"Model precision is: {self.Precision()}")
        print(f"Model F1 score is: {self.F1()}")
        return

    def ConfusionMatrix(self, printfig = True, savefig = False, savefig_path = "cf_plot"):
        cm = confusion_matrix(self.targets, self.predictions)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if savefig: 
            plt.savefig(savefig_path)
        if printfig:
            plt.show(block=False)
        return

    def Accuracy(self):
        return accuracy_score(self.targets, self.predictions)

    def Precision(self):
        return precision_score(self.targets, self.predictions)

    def F1(self):
        return f1_score(self.targets, self.predictions)
    
