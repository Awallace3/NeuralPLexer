import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import math
from matplotlib.pyplot import figure

sns.set()

def plot_eval_casf_graphsage_ds():
    df = pd.read_csv("./outs/eval_casf_graphsage_ds.csv", index_col='i')
    print(df)
    ref_column = "CASF2016"
    model_prediction_columns = [
        # "AffiNETy_graphSage_boltzmann_avg",
        "AffiNETy_graphSage_boltzmann_avg_Q",
        "AffiNETy_graphSage_boltzmann_mlp",
        "AffiNETy_graphSage_boltzmann_mlp2",
        "AffiNETy_graphSage_boltzmann_mlp3",
    ]
    spearman = df.corr()
    print(spearman)

    for n, col in enumerate(model_prediction_columns):
        # Calculate the MAE and RMSE
        true = [np.log10(np.exp(i)) for i in df[ref_column].values]
        pred = [np.log10(np.exp(i)) for i in df[col].values]
        # mae = mean_absolute_error(y_true, y_pred)
        # rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Plot the data
        plt.figure()
        # plt.scatter(y_true, y_pred, edgecolor='k', facecolor='none')
        # plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Diagonal Reference')  # Diagonal reference line
        # plt.legend()

        # Annotate the plot with MAE and RMSE
        # plt.legend([f'MAE: {mae:.3f}', f'RMSE: {rmse:.3f}'])
        # plt.legend([f'MAE: {mae:.3f}', f'RMSE: {rmse:.3f}'])

        plt.plot(pred, true,"r.")
        plt.plot(np.unique(true), np.poly1d(np.polyfit(pred, true, 1))(np.unique(true)))
        plt.text(9., 2.5, "RMSE = "+ str(math.sqrt(mean_squared_error(true, pred)))[:5] )
        plt.text(9., 3., "$R^{2}$ = "+ str(r2_score(true, pred))[:5])
        plt.xlabel('Predicted Constants')
        plt.ylabel('True Constants')

        # Saving plot to file
        plt.savefig(f"./outs/{col}_vs_reference_plot.png")
        plt.close()  # Close the figure to free memory

def plot_eval_casf_torchmd_ds():
    df = pd.read_csv("./outs/eval_casf_torchmd_ds.csv", index_col='i')
    print(df)
    ref_column = "CASF2016"
    model_prediction_columns = [
        "AffiNETy_ViSNet_boltzmann_mlp",
        "AffiNETy_ViSNet_boltzmann_mlp2",
    ]
    spearman = df.corr()
    print(spearman)

    for n, col in enumerate(model_prediction_columns):
        # Calculate the MAE and RMSE
        true = df[ref_column].values
        pred = df[col].values
        # mae = mean_absolute_error(y_true, y_pred)
        # rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Plot the data
        plt.figure()
        # plt.scatter(y_true, y_pred, edgecolor='k', facecolor='none')
        # plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='Diagonal Reference')  # Diagonal reference line
        # plt.legend()

        plt.plot(pred, true,"r.")
        plt.plot(np.unique(true), np.poly1d(np.polyfit(pred, true, 1))(np.unique(true)))
        plt.text(9., 2.5, "RMSE = "+ str(math.sqrt(mean_squared_error(true, pred)))[:5] )
        plt.text(9., 3., "$R^{2}$ = "+ str(r2_score(true, pred))[:5])
        plt.xlabel('Predicted Constants')
        plt.ylabel('True Constants')

        # Saving plot to file
        plt.savefig(f"./outs/{col}_vs_reference_plot.png")
        plt.close()  # Close the figure to free memory

def main():
    plot_eval_casf_graphsage_ds()
    plot_eval_casf_torchmd_ds()

if __name__ == "__main__":
    main()
