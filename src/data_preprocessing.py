import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self , train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train = None
        self.test = None

    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        return self.train, self.test

    def drop_id_columns(self):
        if "Id" in self.train.columns:
            self.train.drop("Id" , axis = 1 ,inplace = True)

        if "Id" in self.test.columns:
            self.test.drop("Id" , axis = 1 ,inplace = True)

    def remove_high_missing_features(self , threshold=0.5):
        """Remove features with more than 50% missing values."""
        missing_ratio = self.train.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        self.train.drop(columns = cols_to_drop , inplace = True)
        self.test.drop(columns = cols_to_drop , inplace = True)

        print(f"Removed High-missing features: {cols_to_drop}")

    def remove_low_variance_features(self):
        """Remove features with very low variance (almost constant)"""
        low_var_cols = [col for col in self.train.columns if self.train[col].nunique() <=1]


        self.train.drop(columns = low_var_cols, inplace = True)
        self.test.drop(columns = low_var_cols , inplace = True)

        print(f"Removed Low-variance features: {low_var_cols}")

    def handle_missing_values(self):

        # Numerical features → fill with median
        num_cols = [col for col in self.train.select_dtypes(include = np.number).columns
                    if col in self.test.columns]

        for col in num_cols:
            median_value = self.train[col].median()
            self.train[col] = self.train[col].fillna(median_value)
            self.test[col] = self.test[col].fillna(median_value)

        # Categorical features → fill with mode
        cat_cols = [col for col in self.train.select_dtypes(include = "object").columns
                    if col in self.test.columns]

        for col in cat_cols:
            mode_value = self.train[col].mode()[0]
            self.train[col] = self.train[col].fillna(mode_value)
            self.test[col] = self.test[col].fillna(mode_value)

    def preprocess(self):
        self.load_data()
        self.drop_id_columns()
        self.remove_high_missing_features()
        self.handle_missing_values()
        self.remove_low_variance_features()

        return self.train, self.test
