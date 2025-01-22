# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from face.kerasface import face_contrib
from face.fnnmodel import FNNModel

import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras.layers import Input, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

import keras

import seaborn as sns

import lime.lime_tabular
import shap
from fcp.fcp import forward_composition

import sys

from face_utils import get_lime_contrib, plot_bar_contrib    


def get_str_val(val, decs=3):
    s = f'{val}'
    if decs == 3:
        if val == 0:
            s = '0.000'
        elif abs(val) < 0.001:
            s = f'{val:.02e}'
        else:
            s = f'{val:.03f}'
    elif decs == 4:
        if val == 0:
            s = '0.0000'
        elif abs(val) < 0.0001:
            s = f'{val:.02e}'
        else:
            s = f'{val:.04f}'
    return s

  

'''
Programa de pruebas de ejecución de la red
'''

# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(1)
       
test_size = 0.20
validation_split = 0.1 
epochs = 60

'''
REGRESSION
'''

n_args = len(sys.argv) 
    
if n_args == 1:
    # ds_name = 'cooling'
    ds_name = 'delta'
else:
    ds_name = sys.argv[1]
    
if ds_name == 'delta':
    delevators_df = pd.read_csv('datasets/delta_elevators.csv', delimiter=';')
    
    X = delevators_df.iloc[:, :-1]
    y = delevators_df.iloc[:, -1].to_numpy()
    
    feature_names = X.columns
    num_inputs =  X.shape[1]
    num_outputs = 1
    
    dataset_name = 'Delta Elevators'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(30, activation='relu')(input_layer)
    hidden_layer = Dense(20, activation='relu')(hidden_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)

    test_size = 0.20
    validation_split = 0.1 
    epochs = 60

    
    model_version = 0
elif ds_name == 'cooling':
    '''
    Energy Efficency - Cooling Load - Regression
    '''
    heating_df = pd.read_csv('datasets/energy_efficency_reg.csv', delimiter=';')
    
    X = heating_df.iloc[:, :-2]
    y = heating_df['Cooling Load'].to_numpy()
    
    feature_names = X.columns
    num_inputs =  X.shape[1]
    num_outputs = 1
    
    dataset_name = 'Energy Efficiency - Cooling Load'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(100, activation='relu')(input_layer)
    hidden_layer = Dense(50, activation='relu')(hidden_layer)
    hidden_layer = Dense(10, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    test_size = 0.20
    validation_split = 0.1 
    epochs = 60
    
    model_version = 0




'''
Escalamos el dataset de entrenamiento
'''

scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    )



'''
Creamos la red con el MLP definido
'''
model = FNNModel(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
                  loss='mean_squared_error')


''' 
Entrenamos la red o utilizamos los pesos de una red preentrenada si existe 
para la version indicada en model_version
'''

print('\nREGRESSION\n')
print(f'Using dataset: {dataset_name}, {len(y)} samples, {num_inputs} features, {num_outputs} output, ({len(y_train)} train / {len(y_test)} test)')


model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)

print('\nGenerating predictions over test data with MLP ... ', end='')
mlp_predictions = model.predict(X_test, verbose=0)
print('OK')

rmse = np.sqrt(metrics.mean_squared_error(y_test, mlp_predictions))
r2 = metrics.r2_score(y_test, mlp_predictions)

print(f'MLP results for test data RMSE = {rmse:.5f}')




sns.set_style('white')

'''
REGRESSION: PIRO Feature Relevance computation method
'''

save_file = False


# Selected sample for Delta
samples = [499]

# Selected sample for Pima
# samples = [10]


for sample in samples:

    y_mlp = mlp_predictions[sample][0]
    
    y_mlp_screen = get_str_val(y_mlp)
    
    print(f'\nPlotting PIRO regression feature relevance for sample {sample} ... ', end='')
    
    piro_contrib = face_contrib(X_test.iloc[sample], model)[0]
    
    y_piro = np.sum(piro_contrib)
    
    y_piro_screen = get_str_val(y_piro)
    
    plot_bar_contrib(feature_names, piro_contrib, mode='regression',
      title=f'Truth/Net/Exp={y_test[sample]:.03f}/{y_mlp_screen}/{y_piro_screen} Intercept={piro_contrib[0]:.03f}',
      save_file=save_file,
      graph_fname=f'individual_regression_graph_output-{dataset_name}_sample_{sample}_rmse_{rmse:.03f}_PIRO.svg')
    
    print('OK')
    
    
    
    '''
    Regression + LIME
    '''
    print(f'Plotting LIME regression feature relevance for sample {sample} ... ', end='')
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                        feature_names=feature_names, 
                                                        mode='regression', 
                                                        discretize_continuous=False,
                                                        verbose=False,
                                                        random_state=33)
                                                      
    
    lime_contrib, lime_prediction, lime_proba = get_lime_contrib(lime_explainer,
                                                        X_test.iloc[sample].to_numpy(),
                                                        model.predict,
                                                        num_inputs, num_outputs,
                                                        return_weighted=True,
                                                        mode='regression')
     
    
    y_lime = lime_prediction[1]
    
    lime_contrib = lime_contrib[1]
    
    y_lime_screen = get_str_val(y_lime)

    plot_bar_contrib(feature_names, lime_contrib, 
                    mode='regression', 
                    title=f'Truth/Net/Exp={y_test[sample]:.03f}/{y_mlp_screen}/{y_lime_screen} Intercept={lime_contrib[0]:.03f}',
                    save_file=save_file,
                    graph_fname=f'individual_regression_graph_output-{dataset_name}_sample_{sample}_rmse_{rmse:.03f}_LIME.svg')
    
    print('OK')
    
    
    ''' 
    FCP
    '''
    
    print(f'Plotting FCP regression feature relenvance for sample {sample} ... ', end='')
    
    
    fcp_contrib = forward_composition(model, X_test.iloc[sample])[-1][0]
    
    y_fcp = np.sum(fcp_contrib)
    
    y_fcp_screen = get_str_val(y_fcp)
    
    fcp_contrib = np.hstack((0, fcp_contrib))
    
    fcp_importance_feature_order = np.argsort(fcp_contrib)
    
    plot_bar_contrib(feature_names, fcp_contrib, 
                     mode='regression', 
                     title=f'Truth/Net/Exp={y_test[sample]:.03f}/{y_mlp_screen}/{y_fcp_screen}',
                     save_file=save_file,
                     graph_fname=f'individual_regression_graph_output-{dataset_name}_sample_{sample}_rmse_{rmse:.03f}_FCP.svg')
    
    print('OK')
    
    
    '''
    Regression + SHAP
    '''
    
    print(f'Plotting SHAP regression feature relevance for sample {sample} ... ', end='')
    
    
    '''
    Cambiamos un ejemplo mean del dataset
    '''
    shap_explainer = shap.KernelExplainer(model.predict, X.mean())

    shap_explanation = shap_explainer(X_test.iloc[sample].to_numpy())
    
    shap_values = shap_explanation.values
    
    y_shap = (np.sum(shap_values) + shap_explainer.expected_value)[0]
    
    y_shap_screen = get_str_val(y_shap)
    
    shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))[0]
    
    plot_bar_contrib(feature_names, shap_contrib, 
                      mode='regression', 
                      title=f'Truth/Net/Exp={y_test[sample]:.03f}/{y_mlp_screen}/{y_shap_screen} Expected value={shap_contrib[0]:.03f}',
                      save_file=save_file,
                      graph_fname=f'individual_regression_graph_output-{dataset_name}_sample_{sample}_rmse_{rmse:.03f}_SHAP.svg'
                      )
    
    
    print('OK')

    print()
    