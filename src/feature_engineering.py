import pandas as pd
from sklearn.preprocessing import LabelEncoder , StandardScaler

class FeatureEngineer:
    def __init__(self , train , test):
        self.train = train
        self.test = test
        self.label_encoder = {}
        self.scaler = StandardScaler()

    def encode_categorical(self):
        """Label Encoding for categorical features"""

        cat_cols = self.train.select_dtypes(include = ["object"]).columns

        for col in cat_cols:
            le = LabelEncoder()
            self.train[col] = le.fit_transform(self.train[col])
            self.test[col] = le.transform(self.test[col])
            self.label_encoder[col] = le

    def scale_numeric(self):
        """Scale numeric features"""
        num_cols = [col for col in self.train.select_dtypes(include = ["int64" , "float64"]).columns
                    if col in self.test.columns
        ]


        self.train[num_cols] = self.scaler.fit_transform(self.train[num_cols])
        self.test[num_cols] = self.scaler.transform(self.test[num_cols])

    def feature_engineer(self):
        """Full feature engineering pipeline"""
        self.encode_categorical()
        self.scale_numeric()
        return self.train, self.test