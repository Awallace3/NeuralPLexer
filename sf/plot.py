import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def plot_eval_casf():
    df = pd.read_csv("./outs/eval_casf.csv")
    ref_column = "CASF2016"
    model_prediction_columns = [
        "AffiNETy_graphSage_boltzmann_avg",
        "AffiNETy_graphSage_boltzmann_avg_Q",
        "AffiNETy_graphSage_boltzmann_mlp",
    ]

    for col in model_prediction_columns:
        # Calculate the MAE and RMSE
        y_true = df[ref_column].values
        y_pred = df[col].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Plot the data
        plt.figure()
        plt.scatter(y_true, y_pred, edgecolor='k', facecolor='none')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)  # Diagonal reference line

        # Annotate the plot with MAE and RMSE
        plt.legend([f'MAE: {mae:.3f}', f'RMSE: {rmse:.3f}'])

        plt.title(f'Predictions vs. {ref_column} Reference')
        plt.xlabel('Reference Values')
        plt.ylabel('Predicted Values')

        # Saving plot to file
        plt.savefig(f"./outs/{col}_vs_reference_plot.png")
        plt.close()  # Close the figure to free memory

def main():
    plot_eval_casf()

if __name__ == "__main__":
    main()
