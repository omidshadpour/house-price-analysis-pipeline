import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

class ModelTrainer:
    def __init__(self , train_df , test_df ,target_col = "SalePrice"):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col

        self.X = self.train_df.drop(self.target_col, axis = 1)
        self.y = self.train_df[self.target_col]

        self.best_models = {}

    def train_test_split(self , test_size = 0.2 , random_state = 42 ):
        x_train, x_valid, y_train, y_valid = train_test_split(
            self.X, self.y, test_size = test_size, random_state = random_state
        )
        return x_train, x_valid, y_train, y_valid

    def evaluate_model(self , model , x_valid , y_valid , model_name = "Model"):
        preds = model.predict(x_valid)
        rmse = np.sqrt(mean_squared_error(y_valid , preds))
        r2 = r2_score(y_valid , preds)
        print(f"{model_name} -> RMSE: {rmse:.4f} , R2: {r2:.4f}")

        return rmse, r2

    def train_linear_regression(self):
        print("\n Training Linear Regression...")
        x_train, x_valid, y_train, y_valid = self.train_test_split()

        lr = LinearRegression()
        lr.fit(x_train, y_train)

        rmse , r2 =self.evaluate_model(lr , x_valid , y_valid , "LinearRegression")

        self.best_models["LinearRegression"] = {
            "model" : lr,
            "rmse" : rmse,
            "r2" : r2
        }

    def train_random_forest(self):
        print("\n Training Random Forest with GridSearchCV...")
        x_train, x_valid, y_train, y_valid = self.train_test_split()

        rf = RandomForestRegressor(random_state = 42)

        param_grid = {
            "n_estimators" : [100, 200],
            "max_depth" : [None , 10, 20],
            "min_samples_split" : [2 , 5]
        }

        grid_search = GridSearchCV(
            estimator = rf ,
            param_grid = param_grid ,
            cv = 3 ,
            scoring = "neg_root_mean_squared_error" ,
            n_jobs = -1 ,
            verbose = 1
        )

        grid_search.fit(x_train , y_train)

        best_rf = grid_search.best_estimator_
        print(f"Best RF Params: {grid_search.best_params_}")

        rmse , r2 = self.evaluate_model(best_rf, x_valid , y_valid , "RandomForest")

        self.best_models["RandomForest"] = {
            "model" : best_rf,
            "rmse" : rmse,
            "r2" : r2
        }

    def train_xgboost(self):
        print("\n Training XGBoost with GridSearchCV...")
        x_train, x_valid, y_train, y_valid = self.train_test_split()

        xgb = XGBRegressor(
            objective = "reg:squarederror" ,
            random_state=42 ,
            tree_method = "hist" ,
        )

        param_grid = {
            "n_estimators" : [200, 400],
            "max_depth" : [3 , 5, 7],
            "learning_rate" : [0.05 , 0.1],
            "subsample" : [0.8 , 1.0]
        }

        grid_search = GridSearchCV(
            estimator = xgb ,
            param_grid = param_grid ,
            cv = 3 ,
            scoring = "neg_root_mean_squared_error" ,
            n_jobs = -1 ,
            verbose = 1
        )

        grid_search.fit(x_train , y_train)

        best_xgb = grid_search.best_estimator_
        print(f"Best XGB Params: {grid_search.best_params_}")

        rmse , r2 = self.evaluate_model(best_xgb, x_valid , y_valid , "XGBoost")

        self.best_models["XGBoost"] = {
            "model" : best_xgb,
            "rmse" : rmse,
            "r2" : r2
        }

    def compare_and_save_best_model(self , save_path = "models/best_model.pkl"):
        results = []

        for name , info  in self.best_models.items():
            results.append((name , info["rmse"] , info["r2"]))

        results_df = pd.DataFrame(results , columns = ["Model" , "rmse" , "r2"])
        print("\nModel Comparison:")
        print(results_df.sort_values(by = "rmse"))

        best_row = results_df.sort_values(by = "rmse").iloc[0]
        best_model_name = best_row["Model"]
        best_model = self.best_models[best_model_name]["model"]

        print(f"\n Best Model: {best_model_name} -> saving to {save_path}")
        joblib.dump(best_model , save_path)

        return best_model_name , best_model , results_df


