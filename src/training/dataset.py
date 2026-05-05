import pandas as pd
from src.config import settings

class FraudDataset:
    def __init__(self, train_path = settings.TRAIN_PATH, val_path = settings.VALIDATION_PATH):
        self.train_path = train_path
        self.val_path = val_path

    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        return train_df, val_df
    
    def split_features_target(self, df):
        X = df.drop(columns = settings.DROP_COLS)
        y = df[settings.TARGET_COL]
        return X, y
    
    def get_train_validation_data(self):
        train_df, val_df = self.load_data()
        X_train, y_train = self.split_features_target(train_df)
        X_val, y_val = self.split_features_target(val_df)
        return X_train, y_train, X_val, y_val