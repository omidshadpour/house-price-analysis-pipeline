import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureImportancePlotter:
    def __init__(self , model , feature_name , save_dir = "reports/figures"):
        self.model = model
        self.feature_name = feature_name
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_importances(self):
        """Extract feature importances from RF or XGB models"""
        importances = self.model.feature_importances_

        df = pd.DataFrame({
            "feature" : self.feature_name,
            "importance" : importances
        }).sort_values(by = "importance", ascending = False)

        return  df

    def plot_importances(self , top_n = 20):
        """Plot top N most important features."""
        df = self.get_importances().head(top_n)

        plt.figure(figsize = (10 , 8))
        sns.barplot(x = df["importance"] , y = df["feature"], hue = df["feature"],palette = "viridis" , legend = False)
        plt.title(f"Top {top_n} Most Important Features")
        plt.tight_layout()

        save_path = f"{self.save_dir}/feature_importance_top{top_n}.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Feature importance plot saved to: {save_path}")
        return df


