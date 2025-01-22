# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

from face.kerasface import face_contrib
from face.kerasface import hard_sigmoid, hard_tanh
from face.fnnmodel import FNNModel

import lime.lime_tabular
import shap
from fcp.fcp import forward_composition

import keras
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_DETERMINISTIC_OPS"] = '1'

from tensorflow.keras.layers import Input, Dense

import time
import pandas as pd
import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from face_utils import get_lime_contrib


'''
REGRESSION TIME
'''


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
Incorporamos las funciones hard_sigmoid y hard_tanh
'''
get_custom_objects().update({'hard_sigmoid': hard_sigmoid})
get_custom_objects().update({'hard_tanh': hard_tanh})


n_args = len(sys.argv) 

if n_args == 1:
    # ds_name = 'cooling'
    ds_name = 'delta'
else:
    ds_name = sys.argv[1]
    

if ds_name == 'delta':
    '''
    Delta Elevators dataset - Regression - 6 feat, 9517 samples
    LIME global = 1h17m
    RMSE 0.00151
    '''
    
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
    validatation_split = 0.10 
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
    validatation_split = 0.1 
    epcohs = 60
    
    model_version = 0



scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    )

n_samples = 10
# n_samples = 1000
save_file = False

print(f'\nCalculating elepased time for {n_samples} samples of {dataset_name}\n')
print(f'Using dataset: {dataset_name}, {len(y_test) + len(y_train)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')

''' 
Creamos y entrenamos el modelo con la definición específica para cada dataset
'''
model = FNNModel(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
                  loss='mean_squared_error')


print('Training regression model ... ', end='')
model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)
print('OK')



'''
Realizamos predicciones y evaluamos el resultado
'''
print('\nGenerating predictions over test data with MLP ... ', end='')
mlp_predictions = model.predict(X_test, verbose=0)
print('OK')

rmse = np.sqrt(metrics.mean_squared_error(y_test, mlp_predictions))
r2 = metrics.r2_score(y_test, mlp_predictions)

print(f'MLP results for test data RMSE = {rmse:.5f}')


'''
Create LIME and kSHAP explainers
'''
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                        feature_names=feature_names, 
                                                        mode='regression', 
                                                        discretize_continuous=False,
                                                        verbose=False,
                                                        random_state=33)

shap_explainer = shap.KernelExplainer(model.predict, X.mean())

shap_explainer_zeros = shap.KernelExplainer(model.predict, np.zeros((1, num_inputs)))



n_shots = 1

samples = np.random.choice(range(len(y_train)), n_samples)

explainer_names = ['FACE', 'LIME', 'kSHAP-mean', 'kSHAP-zero', 'FCP']
# explainer_names = ['FACE']
   
for shot in range(n_shots):
   
    t_dict = dict()
    
    print(f'Timing shot {shot+1} ... ', end='')

    for explainer_name in explainer_names:
        time_list = []
        print(f'Computing time for {explainer_name} on {n_samples} samples ...', end='', flush=True)
        
        for sample in samples:
            t = time.time()
            match explainer_name:
                case 'FACE':
                    piro_contrib = face_contrib(X_train.iloc[sample], model)[0]
        
                case 'LIME':        
                    lime_contrib, lime_prediction, lime_proba = get_lime_contrib(lime_explainer,
                                                                        X_train.iloc[sample].to_numpy(),
                                                                        model.predict,
                                                                        num_inputs, num_outputs,
                                                                        return_weighted=True,
                                                                        mode='regression')
     
                case 'kSHAP-mean':                           
                    shap_explanation = shap_explainer(X_train.iloc[sample].to_numpy())
                    shap_values = shap_explanation.values
                    y_shap = (np.sum(shap_values) + shap_explainer.expected_value)[0]
                    shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))[0]
                    
                case 'kSHAP-zero':                           
                    shap_explanation_zeros = shap_explainer_zeros(X_train.iloc[sample].to_numpy())
                    shap_values_zeros = shap_explanation_zeros.values
                    y_shap_zeros = (np.sum(shap_values_zeros) + shap_explainer_zeros.expected_value)[0]
                    shap_contrib_contrib = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), shap_values_zeros.T))[0]

                case 'FCP':
                    fcp_values = forward_composition(model, X_train.iloc[sample])[-1]
                    fcp_contrib = np.hstack((np.zeros(num_outputs).reshape(-1,1), fcp_values))
    
            t = time.time() - t
            time_list.append(t)
        
        time_array = np.array(time_list)
        
        print(f'done in {np.mean(time_array)}s +/- {np.std(time_array)}')
        t_dict[explainer_name] = (np.mean(time_array), np.std(time_array))
        
        
    cad = f'{ds_name}' 
    for explainer_name in explainer_names:
        cad = f'{cad};{explainer_name};{t_dict[explainer_name][0]};{t_dict[explainer_name][1]}'    
    
    if save_file:    
        with open('Time Output/global_times_accelerated.csv', 'a') as output_file:
            print(cad)
            output_file.write(cad + '\n')
