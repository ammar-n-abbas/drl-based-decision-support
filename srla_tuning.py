import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# data = list(map(pd.read_csv, glob.glob("\Simulator_Logs/*.xlsx")))
# data_length = list(map(len, data))

# data_concat = pd.concat(data, ignore_index=True)

def objective(trial):
    try:
        data_hmm = data_concat

        n_states = trial.suggest_int('n_states', 1, 10)
        model_type = trial.suggest_categorical('model_type', ['hmm', 'gmmhmm'])
        if model_type == 'gmmhmm':
            n_mix = trial.suggest_int('n_mix', 1, 5)
        covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
        n_iter = 200
        tol = 1e-2

        # is_lr = trial.suggest_categorical('is_lr', [True, False])
        is_lr = True
        # is_pca = trial.suggest_categorical('is_pca', [True, False])
        is_pca = True

        if is_pca:
            n_decomp = trial.suggest_int('n_decomp', 1, 20)
            data_hmm = MinMaxScaler().fit_transform(PCA(n_decomp).fit_transform(data_hmm))
        if not is_pca:
            is_scaler = trial.suggest_categorical('is_scaler', [True, False])
            if is_scaler:
                data_hmm = MinMaxScaler().fit_transform(data_hmm)

        if is_lr:
            init_params = "cm"
            params = "cmt"
        else:
            init_params = "stmcw"
            params = "stmcw"

        # Print the chosen hyperparameters
        print(
            f"Chosen Hyperparameters: "
            f"n_states={n_states}, "
            f"model_type={model_type}, "
            f"covariance_type={covariance_type}, "
            f"n_iter={n_iter}, "
            f"tol={tol}, "
            f"is_lr={is_lr}, "
            f"is_pca_decomp={is_pca}, ")
        if model_type == 'gmmhmm':
            print(f"n_mix={n_mix}")
        if is_pca:
            print(f"n_decomp={n_decomp}")
        else:
            print(f"is_scaler={is_scaler}")

        if model_type == 'gmmhmm':
            model = hmm.GMMHMM(
                n_components=n_states,
                n_mix=n_mix,
                covariance_type=covariance_type,
                init_params=init_params,
                params=params,
                n_iter=n_iter,
                tol=tol,
                verbose=True
            )
        else:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                init_params=init_params,
                params=params,
                n_iter=n_iter,
                tol=tol,
                verbose=True
            )

        if is_lr:
            model.startprob_ = np.array([1.0] + [0.0] * (n_states - 1))

            transition_matrix = np.eye(n_states) * 0.5 + np.eye(n_states, k=1) * 0.5
            transition_matrix[-1, -1] = 1.0
            model.transmat_ = transition_matrix

        model.fit(data_hmm, data_length)

        pred = model.predict(data_hmm, data_length).reshape(-1, 1)

        preds = []
        for i in range(len(data_length)):
            if i == 0:
                prediction = pred[0:data_length_cum[i]]
            else:
                prediction = pred[data_length_cum[i - 1]:data_length_cum[i]]
            preds.append(list(prediction[-1]))

        acc = (adjusted_rand_score(labels.flatten(), np.array(preds).flatten()) +
               adjusted_mutual_info_score(labels.flatten(), np.array(preds).flatten())) / 2
        return acc

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return 0.0

data_concat = pd.read_csv('data_concat.csv')
numeric_columns = data_concat.select_dtypes(include=['number']).columns
data_concat = data_concat[numeric_columns].fillna(value=0)
data_length = pd.read_csv('data_length.csv').values.flatten()

labels = pd.read_excel('data_validation.xlsx')['Labels'].values

cumulative_sum = 0
data_length_cum = []
for num in data_length:
    cumulative_sum += num
    data_length_cum.append(cumulative_sum)

study = optuna.create_study(study_name='hmm fine-tuning', direction='maximize')
n_trials = 1e50

try:
    study.optimize(objective, n_trials=n_trials)
except KeyboardInterrupt:
    print("ctrl+c pressed")
    optuna.visualization.matplotlib.plot_optimization_history(study)
    optuna.visualization.matplotlib.plot_intermediate_values(study)
    optuna.visualization.matplotlib.plot_contour(study)
    optuna.visualization.matplotlib.plot_param_importances(study)
    optuna.visualization.matplotlib.plot_timeline(study)
    plt.show(block=True)

    best_params = study.best_params
    best_accuracy = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best Accuracy:", best_accuracy)

optuna.visualization.matplotlib.plot_optimization_history(study)
optuna.visualization.matplotlib.plot_intermediate_values(study)
optuna.visualization.matplotlib.plot_contour(study)
optuna.visualization.matplotlib.plot_param_importances(study)
optuna.visualization.matplotlib.plot_timeline(study)
plt.show(block=True)

best_params = study.best_params
best_accuracy = study.best_value
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

# data_hmm = data_concat
#
# n_states = 2
# model_type = 'hmm'
# n_mix = 5
# covariance_type = 'spherical'
# is_lr = True
# is_scalar = True
# is_pca = True
# n_decomp = 10
#
# if is_pca:
#     scaler = MinMaxScaler()
#     pca = PCA(n_decomp)
#     data_hmm = scaler.fit_transform(pca.fit_transform(data_hmm))
#
# if is_scalar and not is_pca:
#     scaler = MinMaxScaler()
#     data_hmm = scaler.fit_transform(data_hmm)
#
# if is_lr:
#     init_params = "cm"
#     params = "cmt"
#
# else:
#     init_params = "stmcw"
#     params = "stmcw"
#
# if model_type == 'gmmhmm':
#     model = hmm.GMMHMM(n_components=n_states,
#                        n_mix=n_mix,
#                        covariance_type=covariance_type,
#                        init_params=init_params,
#                        params=params,
#                        n_iter=200,
#                        tol=0.01,
#                        verbose=True)
# else:
#     model = hmm.GaussianHMM(n_components=n_states,
#                             covariance_type=covariance_type,
#                             init_params=init_params,
#                             params=params,
#                             n_iter=200,
#                             tol=0.01,
#                             verbose=True)
#
# if is_lr:
#     model.startprob_ = np.array([1.0] + [0.0] * (n_states - 1))
#
#     transition_matrix = np.eye(n_states) * 0.5 + np.eye(n_states, k=1) * 0.5
#     transition_matrix[-1, -1] = 1.0
#     model.transmat_ = transition_matrix
#
# model.fit(data_hmm, data_length)
# pred = model.predict(data_hmm, data_length).reshape(-1, 1)
#
# plt.rcParams['font.size'] = 18  # Default font size
# plt.rcParams['axes.labelsize'] = 18  # Font size of axis labels
# plt.rcParams['axes.titlesize'] = 18  # Font size of plot titles
# plt.rcParams['xtick.labelsize'] = 18  # Font size of x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 18  # Font size of y-axis tick labels
# plt.rcParams['legend.fontsize'] = 18  # Font size of legend
#
# i = 1
# plt.scatter(range(pred[data_length_cum[i - 1]:data_length_cum[i]].shape[0]),
#             pred[data_length_cum[i - 1]:data_length_cum[i]],
#             marker='+', s=5, label='Predicted Values', color='g', alpha=0.4)
# plt.yticks(range(0, 2))
# plt.legend()
# plt.xlabel('Simulation Timestep (sec)')
# plt.ylabel('Hidden State')
# plt.title('Human Failure Detection')
# plt.show()

