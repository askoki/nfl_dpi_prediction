from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score


class ResultMetric:
    accuracy = None
    f1 = None
    recall = None
    precision = None
    auc = None

    def __init__(self, pred_classes: [], real_classes: []):
        self.predicted = pred_classes
        self.true = real_classes
        self.calculate_metrics()

    def calculate_metrics(self):
        self.accuracy = accuracy_score(self.true, self.predicted)
        self.f1 = f1_score(self.true, self.predicted)
        self.recall = recall_score(self.true, self.predicted)
        self.precision = precision_score(self.true, self.predicted)
        try:
            self.auc = roc_auc_score(self.true, self.predicted)
        except ValueError:
            print('### AUC could not be calculated')

    def print_metrics(self, model_name: str):
        print(f'{model_name} accuracy: {self.accuracy}')
        print(f'{model_name} f1: {self.f1}')
        print(f'{model_name} recall score: {self.recall}')
        print(f'{model_name} precision score: {self.precision}')
        print(f'{model_name} AUC score: {self.auc}')

    def get_metrics_dict(self, dict_name: str) -> dict:
        return {
            'name': dict_name,
            'f1': self.f1,
            'accuracy': self.accuracy,
            'auc': self.auc,
            'precision': self.precision,
            'recall': self.recall,
        }
