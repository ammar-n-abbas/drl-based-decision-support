import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
# import optuna

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# data = list(map(pd.read_excel, glob.glob(r"C:\Users\abbas\OneDrive - Software Competence Center Hagenberg GmbH\CISC_SCCH\LL3-DRL\Simulator_Logs\ConcatenatedG4/*.xlsx")))
# data_length = list(map(len, data))

# data_concat = pd.concat(data, ignore_index=True)

data_concat = pd.read_csv('data_concat.csv')
numeric_columns = data_concat.select_dtypes(include=['number']).columns
data_concat = data_concat[numeric_columns].fillna(value=0)
data_length = pd.read_csv('data_length.csv').values.flatten()

data_val = pd.read_excel('data_validation.xlsx')
# data_ai_vs_human = [pd.read_excel(file, header=None) for file in glob.glob(r"C:\Users\abbas\OneDrive - Software Competence Center Hagenberg GmbH\CISC_SCCH\LL3-DRL\Simulator_Logs\srla_vs_human/*.xlsx")]
# data_ai_vs_human_concat = pd.concat(data_ai_vs_human, ignore_index=True)

cumulative_sum = 0
data_length_cum = []
for num in data_length:
    cumulative_sum += num
    data_length_cum.append(cumulative_sum)

data_hmm = data_concat

# Trial 51 finished with value: 0.5908774800904669 and parameters: {'n_states': 3, 'model_type': 'gmmhmm', 'n_mix': 2, 'covariance_type': 'full', 'is_lr': False, 'n_decomp': 3}. Best is trial 51 with value: 0.5908774800904669.
# Trial 174 finished with value: 0.5982298270994988 and parameters: {'n_states': 6, 'model_type': 'gmmhmm', 'n_mix': 2, 'covariance_type': 'full', 'is_lr': False, 'n_decomp': 4}. Best is trial 174 with value: 0.5982298270994988.
# Trial 280 finished with value: 0.6295127368426976 and parameters: {'n_states': 4, 'model_type': 'gmmhmm', 'n_mix': 3, 'covariance_type': 'full', 'is_lr': False, 'n_decomp': 4}. Best is trial 280 with value: 0.6295127368426976.
# Trial 37 finished with value: 0.5700378456037727 and parameters: {'n_states': 4, 'model_type': 'gmmhmm', 'n_mix': 1, 'covariance_type': 'tied', 'n_decomp': 5}. Best is trial 37 with value: 0.5700378456037727.
# Best is trial 248 with value: 0.5701838872915297 and parameters: {'n_states': 3, 'model_type': 'gmmhmm', 'n_mix': 3, 'covariance_type': 'tied', 'n_decomp': 4}.

n_states = 3
model_type = 'gmmhmm'
n_mix = 3
covariance_type = 'tied'
is_lr = True
is_scalar = True
is_pca = True
n_decomp = 4

if is_pca:
    scaler = MinMaxScaler()
    pca = PCA(n_decomp)
    data_hmm = scaler.fit_transform(pca.fit_transform(data_hmm))

if is_scalar and not is_pca:
    scaler = MinMaxScaler()
    data_hmm = scaler.fit_transform(data_hmm)

if is_lr:
    init_params = "cm"
    params = "cmt"

else:
    init_params = "stmcw"
    params = "stmcw"

if model_type == 'gmmhmm':
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=covariance_type,
                       init_params=init_params,
                       params=params,
                       n_iter=200,
                       tol=0.01,
                       verbose=True)
else:
    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type=covariance_type,
                            init_params=init_params,
                            params=params,
                            n_iter=200,
                            tol=0.01,
                            verbose=True)

if is_lr:
    model.startprob_ = np.array([1.0] + [0.0] * (n_states - 1))

    transition_matrix = np.eye(n_states) * 0.5 + np.eye(n_states, k=1) * 0.5
    transition_matrix[-1, -1] = 1.0
    model.transmat_ = transition_matrix

model.fit(data_hmm, data_length)
pred = model.predict(data_hmm, data_length).reshape(-1, 1)

plt.rcParams['font.size'] = 18  # Default font size
plt.rcParams['axes.labelsize'] = 18  # Font size of axis labels
plt.rcParams['axes.titlesize'] = 18  # Font size of plot titles
plt.rcParams['xtick.labelsize'] = 18  # Font size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 18  # Font size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 18  # Font size of legend

i = 1
plt.scatter(range(pred[data_length_cum[i - 1]:data_length_cum[i]].shape[0]),
            pred[data_length_cum[i - 1]:data_length_cum[i]],
            marker='+', s=5, label='Predicted Values', color='g', alpha=0.4)
plt.yticks(range(0, 2))
plt.legend()
plt.xlabel('Simulation Timestep (sec)')
plt.ylabel('Hidden State')
plt.title('Human Failure Detection')
plt.show()

preds = []
for i in range(len(data_length)):
    if i == 0:
        prediction = pred[0:data_length_cum[i]]
    else:
        prediction = pred[data_length_cum[i - 1]:data_length_cum[i]]
    preds.append(list(prediction[-1]))
preds = np.array(preds)