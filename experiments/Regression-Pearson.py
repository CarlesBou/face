# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras.layers import Input, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import keras

import seaborn as sns
import sys

from face.kerasface import face_contrib
from face.fnnmodel import FNNModel

import lime.lime_tabular
import shap
from fcp.fcp import forward_composition

from face_utils import get_lime_contrib



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
    '''
    Delta Elevators dataset - Regression - 6 feat, 9517 samples
    LIME global = 1h17m
    RMSE 0.00151
    '''
    
    delevators_df = pd.read_csv('delta_elevators.csv', delimiter=';')
    
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
    heating_df = pd.read_csv('energy_efficency_reg.csv', delimiter=';')
    
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
    validatation_split = 0.10 
    epcohs = 60
    
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

print('\nREGRESSION - PEARSON\n')
print(f'Using dataset: {dataset_name}, {len(y)} samples, {num_inputs} features, {num_outputs} output, ({len(y_train)} train / {len(y_test)} test)')


'''
Realizamos unas predicciones y evaluamos el resultado
'''
model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=0)
    
print('\nGenerating predictions over test data with MLP ... ', end='')
mlp_predictions = model.predict(X_test, verbose=0)
print('OK')

rmse = np.sqrt(metrics.mean_squared_error(y_test, mlp_predictions))
r2 = metrics.r2_score(y_test, mlp_predictions)

print(f'MLP results for test data RMSE = {rmse:.5f}')

sns.set_style('white')



save_file = False

face_contrib_list = []
lime_contrib_list = []
shap_contrib_list = []
shap_zeros_contrib_list = []
fcp_contrib_list = []


lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                    feature_names=feature_names, 
                                                    mode='regression', 
                                                    discretize_continuous=False,
                                                    verbose=False,
                                                    random_state=33)

'''
Cambiamos a un ejemplo de background calculado como la media del dataset
'''
shap_explainer = shap.KernelExplainer(model.predict, X.mean())

shap_explainer_zeros = shap.KernelExplainer(model.predict, np.zeros((1, num_inputs)))


samples = [x for x in range(len(y_test))]

for sample in samples:

    y_mlp = mlp_predictions[sample][0]
    
    print(f'\Computing PIRO regression feature relevance for sample {sample} ... ', end='')
    
    piro_contrib = face_contrib(X_test.iloc[sample], model)[0]
    
    y_piro = np.sum(piro_contrib)
        
    face_contrib_list.append(piro_contrib)
    
    print('OK')
    
    
    
    '''
    Regression + LIME
    '''
    print(f'Computing LIME regression feature relevance for sample {sample} ... ', end='')
    
    
    lime_contrib, lime_prediction, lime_proba = get_lime_contrib(lime_explainer,
                                                        X_test.iloc[sample].to_numpy(),
                                                        model.predict,
                                                        num_inputs, num_outputs,
                                                        return_weighted=True,
                                                        mode='regression')
     
    
    y_lime = lime_prediction[1]
    
    lime_contrib = lime_contrib[1]
        
    lime_contrib_list.append(lime_contrib)
    
    print('OK')
    
    
    ''' 
    FCP
    '''
    
    print(f'Computing FCP regression feature relenvance for sample {sample} ... ', end='')
    
    fcp_values = forward_composition(model, X_test.iloc[sample])[-1]
    
    fcp_contrib = np.hstack((np.zeros(num_outputs).reshape(-1,1), fcp_values))

    y_fcp = np.sum(fcp_contrib)
        
    fcp_contrib_list.append(fcp_contrib[0])
        
    print('OK')
    
    
    '''
    Regression + SHAP
    '''
    
    print(f'Computing SHAP regression feature relevance for sample {sample} ... ', end='')
    
    shap_explanation = shap_explainer(X_test.iloc[sample].to_numpy())
    
    shap_values = shap_explanation.values
    
    y_shap = (np.sum(shap_values) + shap_explainer.expected_value)[0]
    
    shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))[0]
        
    shap_contrib_list.append(shap_contrib)
    
    print('OK')

    '''
    Regression + SHAP Zeros
    '''
    
    print(f'Computing SHAP regression feature relevance for sample {sample} ... ', end='')
    
    
    shap_explanation = shap_explainer_zeros(X_test.iloc[sample].to_numpy())
    
    shap_values = shap_explanation.values
    
    y_shap = (np.sum(shap_values) + shap_explainer_zeros.expected_value)[0]
    
    shap_contrib_zeros = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), shap_values.T))[0]
        
    shap_zeros_contrib_list.append(shap_contrib_zeros)
    
    print('OK')

    print()
    

pearson_face_lime = []
pearson_face_shap = []
pearson_face_shap_zeros = []
pearson_face_fcp = []


for var in range(num_inputs):
    v_face = []
    v_lime = []
    v_shap = []
    v_shap_zeros = []
    v_fcp = []
    
    for i in range(len(samples)):
        v_face.append(face_contrib_list[i][var+1])
        v_lime.append(lime_contrib_list[i][var+1])
        v_shap.append(shap_contrib_list[i][var+1])
        v_shap_zeros.append(shap_zeros_contrib_list[i][var+1])
        v_fcp.append(fcp_contrib_list[i][var+1])   

    pearson_face_lime.append(pearsonr(v_face, v_lime).statistic)
    pearson_face_shap.append(pearsonr(v_face, v_shap).statistic)
    pearson_face_shap_zeros.append(pearsonr(v_face, v_shap_zeros).statistic)
    pearson_face_fcp.append(pearsonr(v_face, v_fcp).statistic)



file_text = []


cad = f'{ds_name};LIME - Pearson;'
for var in range(num_inputs):
    cad += f'{pearson_face_lime[var]};'
cad += f'{np.nanmean(pearson_face_lime)};{np.nanstd(pearson_face_lime)}'
file_text.append(cad)


cad = f'{ds_name};SHAP - Pearson;'
for var in range(num_inputs):
    cad += f'{pearson_face_shap[var]};'
cad += f'{np.nanmean(pearson_face_shap)};{np.nanstd(pearson_face_shap)}'
file_text.append(cad)

cad = f'{ds_name};SHAP Zeros - Pearson;'
for var in range(num_inputs):
    cad += f'{pearson_face_shap_zeros[var]};'
cad += f'{np.nanmean(pearson_face_shap_zeros)};{np.nanstd(pearson_face_shap_zeros)}'
file_text.append(cad)

cad = f'{ds_name};FCP - Pearson;'
for var in range(num_inputs):
    cad += f'{pearson_face_fcp[var]};'
cad += f'{np.nanmean(pearson_face_fcp)};{np.nanstd(pearson_face_fcp)}'
file_text.append(cad)


var_names = f'{ds_name}-rmse;{rmse};'

for name in feature_names:
    var_names += name + ';'

var_names += 'PearsonMean;PearsonStd'

if save_file:
    with open(f'Pearson - Spearman output/Pearson output_{ds_name}_55.csv', 'w') as output_file:
        output_file.write(var_names.strip(';') + '\n')
        for r in file_text:
            output_file.write(r.strip(';') + '\n')
            print(r)

