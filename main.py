from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_models import ModelTrainer
from src.feature_importance import FeatureImportancePlotter

def main():
    print("Start House Price Analysis Pipeline... ")

    # Step 1: Preprocessing
    preprocessor = DataPreprocessor(train_path = "data/raw/train.csv", test_path = "data/raw/test.csv")
    train , test = preprocessor.preprocess()

    print("Data Preprocessing Completed!")

    # Step 2: Feature Engineering
    fe = FeatureEngineer(train , test)
    train_fe , test_fe = fe.feature_engineer()

    print("Feature Engineering Completed!")

    # Step 3: Model Training & Comparison
    trainer = ModelTrainer(train_fe , test_fe , target_col="SalePrice")
    trainer.train_linear_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()

    best_model_name , best_model , results_df = trainer.compare_and_save_best_model(
        save_path = "models/best_model.pkl",
    )

    print("\nPipeline Finished successfully!")
    print(f"Best Model: {best_model_name}")
    print("\nFinal results:")
    print(results_df)

    # Step 4: Feature Importance
    feature_names = trainer.X.columns
    plotter = FeatureImportancePlotter(best_model , feature_names)
    importance_df = plotter.plot_importances(top_n = 20)

    print("\n Top 20 important features:")
    print(importance_df)

    print("\n Pipeline Finished successfully!")

if __name__ == "__main__":
    main()
