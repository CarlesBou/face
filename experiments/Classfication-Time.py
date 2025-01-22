# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

from face.kerasface import hard_sigmoid, hard_tanh
from face.kerasface import face_contrib
from face.fnnmodel import FNNModel

import shap
from lime.lime_tabular import LimeTabularExplainer
from fcp.fcp import forward_composition


import keras
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_DETERMINISTIC_OPS"] = '1'

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

import time
import pandas as pd
import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from face_utils import get_lime_contrib


'''
TIMING
'''


'''
Programa de pruebas de ejecución de la red
'''
# np.seed = 1
# random.seed = 1
# tf.random.set_seed(1)
keras.utils.set_random_seed(1)


'''
Incorporamos las funciones hard_sigmoid y hard_tanh
'''
get_custom_objects().update({'hard_sigmoid': hard_sigmoid})
get_custom_objects().update({'hard_tanh': hard_tanh})


n_args = len(sys.argv) 

liver_drinks = 7
    
if n_args == 1:
    ds_name = 'liver'
    # ds_name = 'pima'
    # ds_name = 'breast'
    # ds_name = 'german'
    # ds_name = 'MNIST'
else:
    ds_name = sys.argv[1]


if ds_name == 'breast':
    '''
    Breast Cancer dataset - Classification
    '''
    breastc_df = pd.read_csv('datasets/breast_cancer_db.csv')
    
    X = breastc_df.iloc[:, 2:]
    y = breastc_df.iloc[:, 1]
    
    
    y.replace('B', 0, inplace=True)
    y.replace('M', 1, inplace=True)
    
    y = np.array(y, dtype=int)
    
    feature_names = X.columns
    
    num_inputs =  X.shape[1]
    num_outputs = 2
    
    dataset_name = 'Breast Cancer'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(100, activation='relu')(input_layer)
    hidden_layer = Dense(50, activation='relu')(hidden_layer)
    hidden_layer = Dense(25, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    test_size = 0.10
    validation_split = 0.10
    epochs = 30
    
    model_version = 0
    use_saved_model_weights = False

elif ds_name == 'pima':
    '''
    Pima Diabetes dataset - Classification
    '''
    
    diabetes_df = pd.read_csv('datasets/diabetes.csv')
    X = diabetes_df.iloc[:, :-1]
    y = diabetes_df['Outcome']
    y = diabetes_df.iloc[:, -1].to_numpy()
    
    feature_names = X.columns
    num_inputs =  X.shape[1]
    num_outputs = 2
    
    dataset_name = 'Pima Diabetes'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(30, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    test_size = 0.20
    validation_split = 0.10
    epochs = 70
    
    model_version = 0
    use_saved_model_weights = False

elif ds_name == 'german':
    '''
    German credit 
    
    '''
    german_df = pd.read_csv('datasets/german.data.csv', sep=';', header=None)
    
    X = german_df.iloc[:, :-1]
    y = german_df.iloc[:, -1]
    
    X.columns = [f'F{i}' if i not in (8,12) else 'Sex' if i == 8 else 'Age' for i in range(len(X.iloc[0]))]
    
    X_orig = X.copy()
    
    # Recodificación de marital status & sex into sex
    X.loc[(X['Sex'] == 1) | (X['Sex'] == 3) | (X['Sex'] == 4), 'Sex'] = 0
    X.loc[(X['Sex'] == 2) | (X['Sex'] == 5), 'Sex'] = 1
    
    y = pd.Categorical(y).codes
    
    feature_names = X.columns
    num_inputs = X.shape[1]
    num_outputs = 2
    
    dataset_name = 'German credit'

    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(num_inputs * 2, activation='relu')(input_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    test_size = 0.25
    validation_split = 0.10
    epochs = 70
    
    model_version = 0
    use_saved_model_weights = False

elif ds_name == 'liver':
    '''
    Liver disorder - Classification
    
      Attribute information:
       1. mcv	mean corpuscular volume
       2. alkphos	alkaline phosphotase
       3. sgpt	alamine aminotransferase
       4. sgot 	aspartate aminotransferase
       5. gammagt	gamma-glutamyl transpeptidase
       6. drinks	number of half-pint equivalents of alcoholic beverages
                    drunk per day --> Target. Dichotimeze with (as seen in TURNEY-1995)
                    Class 0 as drinks < 3, Class 1 with drinks >= 3
       7. selector  field used to split data into two sets --> Ignored
    
    '''
    
    liver_df = pd.read_csv('datasets/liver_without_duplicates.csv', delimiter=';')
    
    X = liver_df.iloc[:, :-2]
    y = pd.DataFrame(liver_df['drinks'])
    
    # Dichotomize target, number of drinks per day, in two classes
    y.loc[y['drinks'] < liver_drinks] = 0
    y.loc[y['drinks'] >= liver_drinks] = 1
    y = y.values.ravel().astype(int)
    
    feature_names = X.columns
    num_inputs = X.shape[1]
    num_outputs = 2
    
    dataset_name = 'Liver disorder'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(30, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)    
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    test_size = 0.15
    validation_split = 0.10
    epochs = 70
    
    model_version = 0
    use_saved_model_weights = False

elif ds_name == 'MNIST':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    num_inputs = 28 * 28
    num_outputs = 10
    
    dataset_name = 'MNIST'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(num_inputs * 2, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    model_version = 0
    use_saved_model_weights = True
    
    validation_split = 0.10
    epochs = 10


if ds_name != 'MNIST':
    '''
    Escalamos los datos de entrada con un escalado MinMax y mantenemos el 
    escalador para poder utilizar la reversión (scaler.inverse_transform(X))
    '''
    scaler = MinMaxScaler()
    
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        stratify=y)
    
    y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)
    y_test_categorical = to_categorical(y_train, num_outputs).astype(np.float32)

else:
    X_train = X_train.astype(np.float32) / 255. #Transform integer pixel values to [0,1]
    X_train = X_train.reshape(-1, num_inputs)     #Transfor image matrix into vector
    X_test = X_test.astype(np.float32) / 255.   #Transform integer pixel values to [0,1]
    X_test = X_test.reshape(-1, num_inputs)       #Transfor image matrix into vector
        
    y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)
    y_test_categorical = to_categorical(y_train, num_outputs).astype(np.float32)



n_samples = 1000

print(f'\nCalculating elepased time for {n_samples} samples of {dataset_name}\n')
print(f'Using dataset: {dataset_name}, {len(y_test) + len(y_train)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')

''' 
Creamos y entrenamos el modelo con la definición específica para cada dataset
'''
model = FNNModel(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
              metrics=['accuracy'],
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


weights_file_name = f'{dataset_name}_classfication_{model_version}.weights.h5'

if use_saved_model_weights:
    if os.path.isfile(weights_file_name):
        model.load_weights(weights_file_name)
        print(
            f'Using pretrained classification model weights from file {weights_file_name}')
    else:
        print('Training classification model ... ', end='')
        my_fit = model.fit(X_train, y_train_categorical,
                           epochs=epochs, validation_split=validation_split, verbose=0)
        model.save_weights(weights_file_name)
        print('OK')
else:
    print('Training classification model ... ', end='')
    my_fit = model.fit(X_train, y_train_categorical,
                       epochs=epochs, validation_split=validation_split, verbose=0)
    print('OK')





'''
Realizamos predicciones y evaluamos el resultado
'''
print('Generating predictions for test data ... ', end='')
predictions = model.predict(X_test, verbose=0)
predictions_proba = model.predict_proba(X_test, verbose=0)

y_mlp = np.argmax(predictions, axis=1)

print('OK')

accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print(f'Test data accuracy = {accuracy:.5f}\n')



sns.set_style("white")


if ds_name != 'MNIST':
    lime_explainer = LimeTabularExplainer(X_train.to_numpy(),
                                        feature_names=feature_names,
                                        mode='classification',
                                        discretize_continuous=False,
                                        # kernel_width=np.sqrt(num_inputs * 0.95),
                                        # sample_around_instance=True,
                                        verbose=False,
                                        random_state=33
                                        )

    '''
    Cambiamos a un solo ejemplo medio
    '''
    shap_explainer = shap.KernelExplainer(model.predict, X.mean())
    
    shap_explainer_zeros = shap.KernelExplainer(model.predict, np.zeros((1, num_inputs)))
    
else:
    lime_explainer = LimeTabularExplainer(X_train,
                                    # feature_names=feature_names,
                                    mode='classification',
                                    discretize_continuous=False,
                                    # kernel_width=np.sqrt(num_inputs * 0.95),
                                    # sample_around_instance=True,
                                    verbose=False,
                                    random_state=33
                                    )

    shap_explainer = shap.KernelExplainer(model.predict, 
            np.concatenate((X_train, X_test), axis=0).mean(axis=0).reshape((1, num_inputs)))

    '''
    Cambiamos a un solo ejemplo de zeros
    '''
    shap_explainer_zeros = shap.KernelExplainer(model.predict, np.zeros((1, num_inputs)))


y_test_lime = []

save_file = True

n_shots = 1
# n_samples = 1000

samples = np.random.choice(range(len(y_train)), n_samples)

samples = [0]

explainer_names = ['FACE', 'LIME', 'kSHAP-mean', 'kSHAP-zero', 'FCP']
   

explainer_names = ['FACE']

for shot in range(n_shots):
    t_dict = dict()
    
    for explainer_name in explainer_names:
        time_list = []
        for sample in samples:
            t = time.time()
            match explainer_name:
                case 'FACE':
                    if ds_name != 'MNIST':
                        fcontrib = face_contrib(X_train.iloc[sample], model)
                    else:
                        fcontrib = face_contrib(X_train[sample], model)
                case 'LIME':        
                    if ds_name != 'MNIST':
                        lime_contrib, lime_prediction, lime_prediction_proba = \
                            get_lime_contrib(lime_explainer,
                                            X_train.iloc[sample].to_numpy(),
                                            model.predict_proba,
                                            num_inputs, num_outputs,
                                            return_weighted=True)
                    else:
                        lime_contrib, lime_prediction, lime_prediction_proba = \
                            get_lime_contrib(lime_explainer,
                                            X_train[sample],
                                            model.predict_proba,
                                            num_inputs, num_outputs,
                                            return_weighted=True)
                    
                case 'kSHAP-mean':                           
                    if ds_name != 'MNIST':
                        shap_explanation = shap_explainer(X_train.iloc[sample])
                        shap_values = shap_explanation.values
                        shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))
                    else:                                                           
                        shap_explanation = shap_explainer(X_train[sample])
                        shap_values = shap_explanation.values
                        shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))
    
                case 'kSHAP-zero':                           
                    if ds_name != 'MNIST':
                        shap_explanation_zeros = shap_explainer_zeros(X_train.iloc[sample])
                        shap_values_zeros = shap_explanation_zeros.values
                        shap_contrib_zeros = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), shap_values_zeros.T))
                    else:                                                           
                        shap_explanation_zeros = shap_explainer_zeros(X_train[sample])
                        shap_values_zeros = shap_explanation_zeros.values
                        shap_contrib_zeros = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), shap_values_zeros.T))

                case 'FCP':
                    if ds_name != 'MNIST':
                        fcp_values = forward_composition(model, X_train.iloc[sample])[-1]
                        fcp_contrib = np.hstack((np.zeros(num_outputs).reshape(-1,1), fcp_values))
                    else:
                        fcp_values = forward_composition(model, X_train[sample])
                        fcp_contrib = np.hstack((np.zeros(num_outputs).reshape(-1,1), fcp_values[-1]))

        
            t = time.time() - t
            time_list.append(t)
        
        time_array = np.array(time_list)

        print(f'done in {np.mean(time_array):.04f}s +/- {np.std(time_array):.04f}')
        t_dict[explainer_name] = (np.mean(time_array), np.std(time_array))
        
        
    if save_file:
        cad = f'{ds_name}' 
        for explainer_name in explainer_names:
            cad = f'{cad};{explainer_name};{t_dict[explainer_name][0]};{t_dict[explainer_name][1]}'    
        
            
        with open('Time Output-global_times_accelerated.csv', 'a') as output_file:
            print(cad)
            output_file.write(cad + '\n')

