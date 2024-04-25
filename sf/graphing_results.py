import numpy as np
import pandas as pd
df = pd.read_csv('eval_casf.csv')
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import math
from matplotlib.pyplot import figure
sns.set()
pred = df['AffiNETy_graphSage_boltzmann_avg'].tolist()
true = df['CASF2016'].tolist()
#fig, axs = plt.subplots(2, 2,figsize=(12,10))
plt.plot(pred, true,"r.")
plt.title('AffiNETy(GS-Avg)')
plt.plot(np.unique(true), np.poly1d(np.polyfit(pred, true, 1))(np.unique(true)))
plt.text(20., 4., "RMSE = "+ str(math.sqrt(mean_squared_error(true, pred)))[:5] )

plt.text(20., 6., "$R^{2}$ = "+ str(r2_score(true, pred))[:5])
plt.xlabel('Predicted Constants')
plt.ylabel('True Constants')
#plt.set_title('Random Split')
#axs[0, 1].plot(pred2, true2, "r.")
#axs[0, 1].plot(np.unique(true2), np.poly1d(np.polyfit(pred2, true2, 1))(np.unique(true2)))
#axs[0, 1].text(9., 4., "RMSE = "+ str(math.sqrt(mean_squared_error(true2, pred2)))[:5] )
#axs[0, 1].text(9., 6., "$R^{2}$ = "+str(r2_score(true2, pred2))[:5] )
#axs[0, 1].set_title('Protein-Protein Sturctural Similarity Based Split')
#axs[1, 0].plot(pred3, true3, "r.")
#axs[1, 0].plot(np.unique(true3), np.poly1d(np.polyfit(pred3, true3, 1))(np.unique(true3)))
#axs[1, 0].text(9., 4., "RMSE = "+ str(math.sqrt(mean_squared_error(true3, pred3)))[:5])
#axs[1, 0].text(9., 6., "$R^{2}$ = " +str(r2_score(true3, pred3))[:5] )
#axs[1, 0].set_title('Ligand-Ligand Fingerprints Similarity Based Split ')
#axs[1, 1].plot(pred4, true4, "r.")
#axs[1, 1].plot(np.unique(true4), np.poly1d(np.polyfit(pred4, true4, 1))(np.unique(true4)))
#axs[1, 1].text(9., 4., "RMSE = "+  str(math.sqrt(mean_squared_error(true4, pred4)))[:5])
#axs[1, 1].text(9., 6., "$R^{2}$ = "+ str(r2_score(true4, pred4))[:5])
#axs[1, 1].set_title('Complex-Complex Interaction Similarity Based Split')

#for ax in axs.flat:
#    ax.set(xlabel='Predicted Constants', ylabel='True Constants')
#
#for ax in axs.flat:
#    ax.label_outer()
plt.savefig('AffiNETy_gs_avg.png')
