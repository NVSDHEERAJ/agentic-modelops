from lightgbm import LGBMClassifier
from src.config import settings

class ModelTrainer:
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators = 500,
            learning_rate = 0.05,
            num_leaves = 64,
            subsample = 0.8,
            colsample_bytree = 0.8,
            class_weight = "balanced",
            random_state = settings.RANDOM_STATE,
            n_jobs = -1,
        )
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model
    