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
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf


import os
import time
import pandas as pd
import numpy as np
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_DETERMINISTIC_OPS"] = '1'


tf.config.experimental.enable_op_determinism()

from face_utils import get_lime_contrib





'''
PEARSON ANALYSIS
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


test_size = 0.10
validation_split = 0.10
epochs = 30 

n_args = len(sys.argv) 
    
if n_args == 1:
    # ds_name = 'german'
    ds_name = 'breast'
    # ds_name = 'liver'
    # ds_name = 'pima'
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
    
    def get_network_breast(function, leaky_param=0.3):
        
        match function:
            case 'relu' | 'hard_tanh' | 'hard_sigmoid':
                activation = function
            case 'leaky_relu':
                activation = LeakyReLU(leaky_param)
            case _:
                assert(0)
            
        input_layer = Input(shape=(num_inputs,))
        hidden_layer = Dense(100, activation=activation)(input_layer)
        hidden_layer = Dense(50, activation=activation)(hidden_layer)
        hidden_layer = Dense(25, activation=activation)(hidden_layer)
        output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
        
        return input_layer, output_layer

    
    get_network = get_network_breast

    intput_layer, output_layer = get_network('relu')
    
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
    
    feature_names = X.columns
    num_inputs =  X.shape[1]
    num_outputs = 2
    
    dataset_name = 'Pima Diabetes'
    
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(30, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    '''
    Red para PIMA
    '''

    def get_network_pima(function, leaky_param=0.3):
        
        activation = 'relu'
        
        match function:
            case 'relu' | 'hard_tanh' | 'hard_sigmoid':
                activation = function
            case 'leaky_relu':
                activation = LeakyReLU(leaky_param)
            case _:
                assert(0)
            
        input_layer = Input(shape=(num_inputs,))
        hidden_layer = Dense(30, activation=activation)(input_layer)
        hidden_layer = Dense(5, activation=activation)(hidden_layer)
        output_layer = Dense(num_outputs, activation=activation)(hidden_layer)
        
        return input_layer, output_layer

    get_network = get_network_pima
    
    test_size = 0.20
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
                    drunk per day
       7. selector  field used to split data into two sets
    
    '''

    liver_df = pd.read_csv('datasets/liver_without_duplicates.csv', delimiter=';')
    X = liver_df.iloc[:, :-1]
    y = liver_df['selector']
    y = pd.Categorical(y).codes
    
    feature_names = X.columns
    num_inputs = X.shape[1]
    num_outputs = 2
    
    dataset_name = 'Liver disorder'
    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(30, activation='relu')(input_layer)
    hidden_layer = Dense(5, activation='relu')(hidden_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
    '''
    Red para Liver
    '''

    def get_network_liver(function, leaky_param=0.3):
        
        activation = 'relu'
        
        match function:
            case 'relu' | 'hard_tanh' | 'hard_sigmoid':
                activation = function
            case 'leaky_relu':
                activation = LeakyReLU(leaky_param)
            case _:
                assert(0)
            
        input_layer = Input(shape=(num_inputs,))
        hidden_layer = Dense(30, activation=activation)(input_layer)
        hidden_layer = Dense(5, activation=activation)(hidden_layer)
        output_layer = Dense(num_outputs, activation=activation)(hidden_layer)
        
        return input_layer, output_layer

    get_network = get_network_liver
    
    test_size = 0.15
    validation_split = 0.10
    epochs = 70
    
    model_version = 0
    use_saved_model_weights = False
    
elif ds_name == 'german-numeric':
    '''
    German numeric credit 
    
      Attribute information:
       9. Age
    
    '''
    german_df = pd.read_csv('datasets/german.data-numeric.csv', sep=';', header=None)
    
    X = german_df.iloc[:, :-1]
    y = german_df.iloc[:, -1]
    
    X.columns = [f'F{i}' if i not in (6,9) else 'Sex' if i == 6 else 'Age' for i in range(len(X.iloc[0]))]
    
    # Recodificación de marital status & sex into sex
    X.loc[(X['Sex'] == 1) | (X['Sex'] == 3) | (X['Sex'] == 4), 'Sex'] = 0
    X.loc[(X['Sex'] == 2) | (X['Sex'] == 5), 'Sex'] = 1
    
    X_orig = X.copy()
    
    y = pd.Categorical(y).codes
    
    feature_names = X.columns
    num_inputs = X.shape[1]
    num_outputs = 2
    
    dataset_name = 'German credit'

    
    input_layer = Input(shape=(num_inputs,))
    hidden_layer = Dense(num_inputs * 2, activation='relu')(input_layer)
    output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
    
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
    
    def get_network_MNIST(function, leaky_param=0.3):
        
        match function:
            case 'relu' | 'hard_tanh' | 'hard_sigmoid':
                activation = function
            case 'leaky_relu':
                activation = LeakyReLU(leaky_param)
            case _:
                assert(0)
            
        input_layer = Input(shape=(num_inputs,))
        hidden_layer = Dense(num_inputs * 2, activation=activation)(input_layer)
        hidden_layer = Dense(5, activation=activation)(hidden_layer)
        output_layer = Dense(num_outputs, activation=activation)(hidden_layer)
        
        print(f'ACTIVATTION = {activation}')
        
        return input_layer, output_layer

    get_network = get_network_MNIST
    
    model_version = 0
    use_saved_model_weights = True
    
    validation_split = 0.10
    epochs = 10

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
    
    '''
    Red para German
    '''

    def get_network_german(function, leaky_param=0.3):
        
        activation = 'relu'
        
        match function:
            case 'relu' | 'hard_tanh' | 'hard_sigmoid':
                activation = function
            case 'leaky_relu':
                activation = LeakyReLU(leaky_param)
            case _:
                assert(0)
            
        input_layer = Input(shape=(num_inputs,))
        hidden_layer = Dense(num_inputs * 2, activation=activation)(input_layer)
        output_layer = Dense(num_outputs, activation='linear')(hidden_layer)
        
        return input_layer, output_layer

    get_network = get_network_german

    test_size = 0.25
    validation_split = 0.10
    epochs = 70
        
    model_version = 0
    use_saved_model_weights = False


else:
    assert('No sé a que dataset te refieres')
    
    
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

    activation = 'relu'
    
    weights_file_name = f'{dataset_name}_classfication_{model_version}_{activation}_weights.h5'

    model = FNNModel(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='nadam',
                  metrics=['accuracy'],
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    
    if use_saved_model_weights:
        if os.path.isfile(weights_file_name):
            model.load_weights(weights_file_name)
            print(
                f'Using pretrained classification model weights from file {weights_file_name}')
        else:
            print(f'Not found file {weights_file_name}')
            print(f'Training and saving classification model using {activation} ... ', end='')
            my_fit = model.fit(X_train, y_train_categorical,
                               epochs=epochs, validation_split=validation_split, verbose=0)
            model.save_weights(weights_file_name)
            print('OK')
    else:
        print('Training classification model ... ', end='')
        
        my_fit = model.fit(X_train, y_train_categorical,
                           epochs=epochs, validation_split=validation_split, verbose=0)

            
        print('OK')


activation_map = {'relu': 'ReLU', 'leaky_relu': 'Leaky ReLU', 
                  'hard_sigmoid': 'Hard Sigmoid', 'hard_tanh': 'Hard Tanh'}

file_text = []
save_file = True 


if ds_name != 'MNIST':
    n_splits = 10

    # for activation_function in ['relu', 'leaky_relu', 'hard_sigmoid', 'hard_tanh']:    
    for activation_function in ['relu']:    
        train_indx_list = [] 
        test_indx_list = [] 
        y_train_categorical_list = []
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
                
        mlp_accuracy_list = []
        piro_accuracy_list = []
        lime_accuracy_list = []
        fcp_accuracy_list = []
        shap_accuracy_list = []
        shap_zeros_accuracy_list = []
        deepshap_accuracy_list = []
        
        mlp_time_list = []
        piro_time_list = []
        lime_time_list = []
        fcp_time_list = []
        shap_time_list = []
        shap_zeros_time_list = []
        deepshap_time_list = []
        
        piro_kappa_list = []
        lime_kappa_list = []
        fcp_kappa_list = []
        shap_kappa_list = []
        shap_zeros_kappa_list = []
        deepshap_kappa_list = []
        
        cm = dict()
        
        print(f'\nMultiKappa CLASSIFICATION for Activation Function [{activation_function}]\n')
        
        print(
            f'Using dataset: {dataset_name}, {len(y)} samples, {num_inputs} features, {num_outputs} classes')
        
        input_layer, output_layer = get_network(activation_function)
            
        split = 0
        
        for train_indx, test_indx in skf.split(X, y):
            X_train, X_test, y_train, y_test = \
                X.iloc[train_indx], X.iloc[test_indx], y[train_indx], y[test_indx]
            
            y_train_categorical = to_categorical(y_train, num_outputs).astype(np.float32)
        
            ''' 
            Creamos y entrenamos el modelo con la definición específica para cada dataset
            '''
            model = FNNModel(inputs=input_layer, outputs=output_layer)
            
            model.compile(optimizer='nadam',
                          metrics=['accuracy'],
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
            
            print(f'Training FNN {split+1}/{n_splits} ... ', end='')
        
            my_fit = model.fit(X_train, y_train_categorical, epochs=epochs, verbose=0)
            
            print('OK')
            
            t = time.time()
            
            '''
            Realizamos predicciones y evaluamos el resultado
            '''
            print('  Generating FNN  predictions for test data ... ', end='')
            predictions = model.predict(X_test, verbose=0)
            predictions_proba = model.predict_proba(X_test, verbose=0)
            
            y_mlp_test = np.argmax(predictions, axis=1)
            
            mlp_accuracy = np.sum(y_mlp_test == y_test) / len(y_test)
            mlp_accuracy_list.append(mlp_accuracy)
        
            t = time.time() - t 
            mlp_time_list.append(t / len(y_test))
            
            print(f'OK - FNN  Test data accuracy = {mlp_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            
            '''
            PIRO computing
            '''
            
            piro_predictions = []
            
            print('  Generating PIRO predictions for test data ... ', end='')
            
            t = time.time()
            
            for i in range(len(X_test)):
                piro_contrib = face_contrib(X_test.iloc[i], model)
                
                y_piro = np.sum(piro_contrib, axis=1)
                
                piro_predictions.append(y_piro)
                                
            piro_predictions = np.array(piro_predictions)
            
            y_piro_test = np.argmax(piro_predictions, axis=1)
        
            piro_accuracy = np.sum(y_piro_test == y_test) / len(y_test)
            piro_accuracy_list.append(piro_accuracy)
        
            t = time.time() - t 
            piro_time_list.append(t / len(y_test))
            
            print(f'OK - PIRO Test data accuracy = {piro_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            piro_kappa = cohen_kappa_score(y_piro_test, y_mlp_test)
        
            piro_kappa_list.append(piro_kappa)
    

            '''
            LIME computing
            '''
            
            print('  Generating LIME predictions for test data ... ', end='')
            
            t = time.time()
            
            lime_explainer = LimeTabularExplainer(X_train.to_numpy(),
                                                  feature_names=feature_names,
                                                  mode='classification',
                                                  discretize_continuous=False,
                                                  #kernel_width=kernel_width,
                                                  # sample_around_instance=True,
                                                  verbose=False,
                                                  random_state=33)
            lime_predictions = []
            
            for i in range(len(X_test)):
                lime_contrib, lime_prediction, lime_prediction_proba = \
                    get_lime_contrib(lime_explainer, X_test.iloc[i], model.predict_proba, 
                                      num_inputs, num_outputs, return_weighted=True)
                    
                lime_predictions.append(lime_prediction)
                
            y_lime_test = np.array(lime_predictions)
        
            lime_accuracy = np.sum(y_lime_test == y_test) / len(y_test)
            lime_accuracy_list.append(lime_accuracy)
        
            t = time.time() - t 
            lime_time_list.append(t / len(y_test))
            
            print(f'OK - LIME Test data accuracy = {lime_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            lime_kappa = cohen_kappa_score(y_lime_test, y_mlp_test)
        
            lime_kappa_list.append(lime_kappa)
    
            
            '''
            FCP
            '''
            
            fcp_predictions = []
            
            print('  Generating FCP  predictions for test data ... ', end='')
            
            t = time.time()
        
            for i in range(len(X_test)):
                fcp_predictions.append(np.sum(forward_composition(model, X_test.iloc[i])[-1], axis=1))
                
            fcp_predictions = np.array(fcp_predictions)
        
            y_fcp_test = np.argmax(fcp_predictions, axis=1)
        
            fcp_accuracy = np.sum(y_fcp_test == y_test) / len(y_test)
            fcp_accuracy_list.append(fcp_accuracy)
        
            t = time.time() - t 
            fcp_time_list.append(t / len(y_test))
            
            print(f'OK - FCP  Test data accuracy = {fcp_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            fcp_kappa = cohen_kappa_score(y_fcp_test, y_mlp_test)
        
            fcp_kappa_list.append(fcp_kappa)
            
        
            '''
            SHAP
            '''
        
            print('  Generating SHAP predictions for test data ... ', end='')
        
            t = time.time()
            
            shap_predictions = []
    
            '''
            Utilizamos ahora un único ejemplo de background, pero con el valor medio
            '''
            shap_explainer = shap.KernelExplainer(model.predict, X.mean())
            
                                            
            for i in range(len(X_test)):
                shap_values = shap_explainer.shap_values(X_test.iloc[i])
                
                shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), 
                                          shap_values))
        
                shap_y = np.argmax(np.sum(shap_contrib, axis=1))
                
                shap_predictions.append(shap_y)
                
            y_shap_test = np.array(shap_predictions)
        
            shap_accuracy = np.sum(y_shap_test == y_test) / len(y_test)
            shap_accuracy_list.append(shap_accuracy) 
            
            t = time.time() - t 
            shap_time_list.append(t / len(y_test))
            
            print(f'OK - SHAP Test data accuracy = {shap_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
    
            shap_kappa = cohen_kappa_score(y_shap_test, y_mlp_test)
        
            shap_kappa_list.append(shap_kappa)
        
    
            '''
            SHAP-Zeros
            '''
        
            print('  Generating SHAP-Zeros predictions for test data ... ', end='')
        
            t = time.time()
            
            shap_zeros_predictions = []
    
            shap_explainer_zeros = shap.KernelExplainer(model.predict,
                                                  np.zeros((1, num_inputs)))
                                                  # shap.sample(X_train, min(100, len(X_train))),)
        
            for i in range(len(X_test)):
                shap_values = shap_explainer_zeros.shap_values(X_test.iloc[i])
                
                shap_zeros_contrib = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), 
                                          shap_values))
        
                shap_y = np.argmax(np.sum(shap_zeros_contrib, axis=1))
                
                shap_zeros_predictions.append(shap_y)
                
            y_shap_zeros_test = np.array(shap_zeros_predictions)
        
            shap_zeros_accuracy = np.sum(y_shap_zeros_test == y_test) / len(y_test)
            shap_zeros_accuracy_list.append(shap_zeros_accuracy) 
            
            t = time.time() - t 
            shap_zeros_time_list.append(t / len(y_test))
            
            print(f'OK - SHAP-Zeros Test data accuracy = {shap_zeros_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            shap_zeros_kappa = cohen_kappa_score(y_shap_zeros_test, y_mlp_test)
        
            shap_zeros_kappa_list.append(shap_zeros_kappa)
        
            
            split = split + 1
            
        print(f'MLP  - Mean accuracy = {np.mean(mlp_accuracy_list):.4f} ({np.std(mlp_accuracy_list):.4f})')
        file_text.append(f'{activation_map[activation_function]};MLP;{np.mean(mlp_accuracy_list)};{np.std(mlp_accuracy_list)};0;0;{np.mean(mlp_time_list)};{np.std(mlp_time_list)}')
        
        print(f'PIRO - Mean accuracy = {np.mean(piro_accuracy_list):.4f} ({np.std(piro_accuracy_list):.4f})')
        print(f'PIRO - Kappa score = {np.mean(piro_kappa_list):.4f} ({np.std(piro_kappa_list):.4f})')
        print(f'PIRO - Mean prediction time = {np.mean(piro_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};PIRO;{np.mean(piro_accuracy_list)};{np.std(piro_accuracy_list)};{np.mean(piro_kappa_list)};{np.std(piro_kappa_list)};{np.mean(piro_time_list)};{np.std(piro_time_list)}')
        
        print(f'LIME - Mean accuracy = {np.mean(lime_accuracy_list):.4f} ({np.std(lime_accuracy_list):.4f})')
        print(f'LIME - Kappa score = {np.mean(lime_kappa_list):.4f} ({np.std(lime_kappa_list):.4f})')
        print(f'LIME - Mean prediction time = {np.mean(lime_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};LIME;{np.mean(lime_accuracy_list)};{np.std(lime_accuracy_list)};{np.mean(lime_kappa_list)};{np.std(lime_kappa_list)};{np.mean(lime_time_list)};{np.std(lime_time_list)}')
        
        print(f'SHAP - Mean accuracy = {np.mean(shap_accuracy_list):.4f} ({np.std(shap_accuracy_list):.4f})')
        print(f'SHAP - Kappa score = {np.mean(shap_kappa_list):.4f} ({np.std(shap_kappa_list):.4f})')
        print(f'SHAP - Mean prediction time = {np.mean(shap_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};SHAP;{np.mean(shap_accuracy_list)};{np.std(shap_accuracy_list)};{np.mean(shap_kappa_list)};{np.std(shap_kappa_list)};{np.mean(shap_time_list)};{np.std(shap_time_list)}')
        
        print(f'SHAP-Zeros - Mean accuracy = {np.mean(shap_zeros_accuracy_list):.4f} ({np.std(shap_zeros_accuracy_list):.4f})')
        print(f'SHAP-Zeros - Kappa score = {np.mean(shap_zeros_kappa_list):.4f} ({np.std(shap_zeros_kappa_list):.4f})')
        print(f'SHAP-Zeros - Mean prediction time = {np.mean(shap_zeros_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};SHAP-Zeros;{np.mean(shap_zeros_accuracy_list)};{np.std(shap_zeros_accuracy_list)};{np.mean(shap_zeros_kappa_list)};{np.std(shap_zeros_kappa_list)};{np.mean(shap_zeros_time_list)};{np.std(shap_zeros_time_list)}')
        
        print(f'FCP  - Mean accuracy = {np.mean(fcp_accuracy_list):.4f} ({np.std(fcp_accuracy_list):.4f})')
        print(f'FCP  - Kappa score = {np.mean(fcp_kappa_list):.4f} ({np.std(fcp_kappa_list):.4f})')
        print(f'FCP  - Mean prediction time = {np.mean(fcp_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};FCP;{np.mean(fcp_accuracy_list)};{np.std(fcp_accuracy_list)};{np.mean(fcp_kappa_list)};{np.std(fcp_kappa_list)};{np.mean(fcp_time_list)};{np.std(fcp_time_list)}')
            
            
else:
    
    lime_explainer = LimeTabularExplainer(X_train,
                                    mode='classification',
                                    discretize_continuous=False,
                                    verbose=False,
                                    random_state=33)

    shap_explainer = shap.KernelExplainer(model.predict, 
          np.concatenate((X_train, X_test), axis=0).mean(axis=0).reshape((1, num_inputs)))
    
    shap_explainer_zeros = shap.KernelExplainer(model.predict, np.zeros((1, num_inputs)))
    
    X_test_orig = X_test.copy()
    y_test_orig = y_test.copy()
    
    samples = np.random.choice(len(X_test), size=1000, replace=False)
    samples_list = [samples]
    
    # for activation_function in ['relu', 'leaky_relu', 'hard_sigmoid', 'hard_tanh']:    
    for activation_function in ['hard_sigmoid']:    
        mlp_accuracy_list = []
        piro_accuracy_list = []
        lime_accuracy_list = []
        fcp_accuracy_list = []
        shap_accuracy_list = []
        shap_zeros_accuracy_list = []
        deepshap_accuracy_list = []
        
        mlp_time_list = []
        piro_time_list = []
        lime_time_list = []
        fcp_time_list = []
        shap_time_list = []
        shap_zeros_time_list = []
        deepshap_time_list = []
        
        piro_kappa_list = []
        lime_kappa_list = []
        fcp_kappa_list = []
        shap_kappa_list = []
        shap_zeros_kappa_list = []
        deepshap_kappa_list = []
        
        
        weights_file_name = f'{dataset_name}_classfication_{model_version}_{activation_function}_weights.h5'

        input_layer, output_layer = get_network(activation_function)
        
        model = FNNModel(inputs=input_layer, outputs=output_layer)
        
        model.compile(optimizer='nadam',
                      metrics=['accuracy'],
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
        
        use_saved_model_weights = False
        
        if use_saved_model_weights:
            if os.path.isfile(weights_file_name):
                model.load_weights(weights_file_name)
                if activation_function != 'relu':
                    print(
                        f'Using pretrained classification model weights from file {weights_file_name}')
            else:
                print('Training and saving classification model in file {weights_file_name} ... ', end='')
                my_fit = model.fit(X_train, y_train_categorical,
                                    epochs=epochs, validation_split=validation_split, verbose=0)
                model.save_weights(weights_file_name)
                print('OK')
        else:
            print('Training classification model ... ', end='')
            
            my_fit = model.fit(X_train, y_train_categorical,
                                epochs=epochs, validation_split=validation_split, verbose=0)

                
        for split, sample_list_elem in enumerate(samples_list):
            X_test = X_test_orig[sample_list_elem]
            y_test = y_test_orig[sample_list_elem]
            print(f'\nMultiKappa CLASSIFICATION for Activation Function [{activation_function}]\n')
            
            print(
                f'Using dataset: {dataset_name}, {len(X_train) + len(X_test)} samples, {num_inputs} features, {num_outputs} classes')
            
            t = time.time()
            
            '''
            Realizamos predicciones y evaluamos el resultado
            '''
            print('  Generating FNN  predictions for test data ... ', end='')
            predictions = model.predict(X_test, verbose=0)
            predictions_proba = model.predict_proba(X_test, verbose=0)
            
            y_mlp_test = np.argmax(predictions, axis=1)
            
            mlp_accuracy = np.sum(y_mlp_test == y_test) / len(y_test)
            mlp_accuracy_list.append(mlp_accuracy)
        
            t = time.time() - t 
            mlp_time_list.append(t / len(y_test))
            
            print(f'OK - FNN  Test data accuracy = {mlp_accuracy:.6f} - Time={t / len(y_test):.05f}s per predicion for {len(y_test)} predictions')
            
            
            '''
            PIRO computing
            '''
            
            piro_predictions = []
            
            print('  Generating PIRO predictions for test data ... ', end='')
            
            t = time.time()
            
            for sample in range(len(X_test)):
                piro_contrib = face_contrib(X_test[sample], model, return_weighted=True)
                y_piro = np.sum(piro_contrib, axis=1)
                piro_predictions.append(y_piro)            
                
            piro_predictions = np.array(piro_predictions)
            y_piro_test = np.argmax(piro_predictions, axis=1)
        
            piro_accuracy = np.sum(y_piro_test == y_test) / len(y_test)
            piro_accuracy_list.append(piro_accuracy)
        
            t = time.time() - t 
            piro_time_list.append(t / len(y_test))
            
            print(f'OK - PIRO Test data accuracy = {piro_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            piro_kappa = cohen_kappa_score(y_piro_test, y_mlp_test)
            piro_kappa_list.append(piro_kappa)
    
            assert(0)
            
            '''
            LIME computing
            '''            
            print('  Generating LIME predictions for test data ... ', end='')
            
            t = time.time()
            
            lime_predictions = []
            
            for sample in range(len(X_test)):
                lime_contrib, lime_prediction, lime_prediction_proba = \
                    get_lime_contrib(lime_explainer, X_test[sample], model.predict_proba, 
                                      num_inputs, num_outputs, return_weighted=True)
                    
                lime_predictions.append(lime_prediction)
                
            y_lime_test = np.array(lime_predictions)
        
            lime_accuracy = np.sum(y_lime_test == y_test) / len(y_test)
            lime_accuracy_list.append(lime_accuracy)
        
            t = time.time() - t 
            lime_time_list.append(t / len(y_test))
            
            print(f'OK - LIME Test data accuracy = {lime_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            lime_kappa = cohen_kappa_score(y_lime_test, y_mlp_test)
        
            lime_kappa_list.append(lime_kappa)
    
    
            '''
            FCP
            '''
            
            fcp_predictions = []
            
            print('  Generating FCP  predictions for test data ... ', end='')
            
            t = time.time()
        
            for sample in range(len(X_test)):
                fcp_predictions.append(np.sum(forward_composition(model, X_test[sample])[-1], axis=1))
                
            fcp_predictions = np.array(fcp_predictions)
            y_fcp_test = np.argmax(fcp_predictions, axis=1)
        
            fcp_accuracy = np.sum(y_fcp_test == y_test) / len(y_test)
            fcp_accuracy_list.append(fcp_accuracy)
        
            t = time.time() - t 
            fcp_time_list.append(t / len(y_test))
            
            print(f'OK - FCP  Test data accuracy = {fcp_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            fcp_kappa = cohen_kappa_score(y_fcp_test, y_mlp_test)
            fcp_kappa_list.append(fcp_kappa)
    
        
            '''
            SHAP
            '''        
            print('  Generating SHAP predictions for test data ... ', end='')
        
            t = time.time()
            
            shap_predictions = []
    
            '''
            Utilizamos ahora un único ejemplo de background, pero con el valor medio
            '''
                                            
            for sample in range(len(X_test)):
                shap_values = shap_explainer.shap_values(X_test[sample])
                shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), 
                                          shap_values))
        
                shap_y = np.argmax(np.sum(shap_contrib, axis=1))
                shap_predictions.append(shap_y)
                
            y_shap_test = np.array(shap_predictions)
        
            shap_accuracy = np.sum(y_shap_test == y_test) / len(y_test)
            shap_accuracy_list.append(shap_accuracy) 
            
            t = time.time() - t 
            shap_time_list.append(t / len(y_test))
            
            print(f'OK - SHAP Test data accuracy = {shap_accuracy:.6f} - Time={t / len(y_test):.05f}s')
    
            shap_kappa = cohen_kappa_score(y_shap_test, y_mlp_test)
        
            shap_kappa_list.append(shap_kappa)
        
    
            '''
            SHAP-Zeros
            '''
        
            print('  Generating SHAP-Zeros predictions for test data ... ', end='')
        
            t = time.time()
            
            shap_zeros_predictions = []
        
            for sample in range(len(X_test)):
                shap_values = shap_explainer_zeros.shap_values(X_test[sample])
                
                shap_zeros_contrib = np.hstack((shap_explainer_zeros.expected_value.reshape(-1,1), 
                                          shap_values))
        
                shap_y = np.argmax(np.sum(shap_zeros_contrib, axis=1))
                
                shap_zeros_predictions.append(shap_y)
                
            y_shap_zeros_test = np.array(shap_zeros_predictions)
        
            shap_zeros_accuracy = np.sum(y_shap_zeros_test == y_test) / len(y_test)
            shap_zeros_accuracy_list.append(shap_zeros_accuracy) 
            
            t = time.time() - t 
            shap_zeros_time_list.append(t / len(y_test))
            
            print(f'OK - SHAP-Zeros Test data accuracy = {shap_zeros_accuracy:.6f} - Time={t / len(y_test):.05f}s')
            
            shap_zeros_kappa = cohen_kappa_score(y_shap_zeros_test, y_mlp_test)
        
            shap_zeros_kappa_list.append(shap_zeros_kappa)
            

        print(f'MLP  - Mean accuracy = {np.mean(mlp_accuracy_list):.4f} ({np.std(mlp_accuracy_list):.4f})')
        file_text.append(f'{activation_map[activation_function]};MLP;{np.mean(mlp_accuracy_list)};{np.std(mlp_accuracy_list)};0;0;{np.mean(mlp_time_list)};{np.std(mlp_time_list)}')
        
        print(f'PIRO - Mean accuracy = {np.mean(piro_accuracy_list):.4f} ({np.std(piro_accuracy_list):.4f})')
        print(f'PIRO - Kappa score = {np.mean(piro_kappa_list):.4f} ({np.std(piro_kappa_list):.4f})')
        print(f'PIRO - Mean prediction time = {np.mean(piro_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};PIRO;{np.mean(piro_accuracy_list)};{np.std(piro_accuracy_list)};{np.mean(piro_kappa_list)};{np.std(piro_kappa_list)};{np.mean(piro_time_list)};{np.std(piro_time_list)}')
        
        print(f'LIME - Mean accuracy = {np.mean(lime_accuracy_list):.4f} ({np.std(lime_accuracy_list):.4f})')
        print(f'LIME - Kappa score = {np.mean(lime_kappa_list):.4f} ({np.std(lime_kappa_list):.4f})')
        print(f'LIME - Mean prediction time = {np.mean(lime_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};LIME;{np.mean(lime_accuracy_list)};{np.std(lime_accuracy_list)};{np.mean(lime_kappa_list)};{np.std(lime_kappa_list)};{np.mean(lime_time_list)};{np.std(lime_time_list)}')
        
        print(f'SHAP - Mean accuracy = {np.mean(shap_accuracy_list):.4f} ({np.std(shap_accuracy_list):.4f})')
        print(f'SHAP - Kappa score = {np.mean(shap_kappa_list):.4f} ({np.std(shap_kappa_list):.4f})')
        print(f'SHAP - Mean prediction time = {np.mean(shap_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};SHAP;{np.mean(shap_accuracy_list)};{np.std(shap_accuracy_list)};{np.mean(shap_kappa_list)};{np.std(shap_kappa_list)};{np.mean(shap_time_list)};{np.std(shap_time_list)}')
        
        print(f'SHAP-Zeros - Mean accuracy = {np.mean(shap_zeros_accuracy_list):.4f} ({np.std(shap_zeros_accuracy_list):.4f})')
        print(f'SHAP-Zeros - Kappa score = {np.mean(shap_zeros_kappa_list):.4f} ({np.std(shap_zeros_kappa_list):.4f})')
        print(f'SHAP-Zeros - Mean prediction time = {np.mean(shap_zeros_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};SHAP-Zeros;{np.mean(shap_zeros_accuracy_list)};{np.std(shap_zeros_accuracy_list)};{np.mean(shap_zeros_kappa_list)};{np.std(shap_zeros_kappa_list)};{np.mean(shap_zeros_time_list)};{np.std(shap_zeros_time_list)}')
    
        print(f'FCP  - Mean accuracy = {np.mean(fcp_accuracy_list):.4f} ({np.std(fcp_accuracy_list):.4f})')
        print(f'FCP  - Kappa score = {np.mean(fcp_kappa_list):.4f} ({np.std(fcp_kappa_list):.4f})')
        print(f'FCP  - Mean prediction time = {np.mean(fcp_time_list):.4f}s')
        file_text.append(f'{activation_map[activation_function]};FCP;{np.mean(fcp_accuracy_list)};{np.std(fcp_accuracy_list)};{np.mean(fcp_kappa_list)};{np.std(fcp_kappa_list)};{np.mean(fcp_time_list)};{np.std(fcp_time_list)}')
                

if save_file:
    print(f'Writing output file Multi Kappa Output-MultiKappa-Classification-Full-Output_{ds_name}_111.csv')
    with open(f'Multi Kappa Output-MultiKappa-Classification-Full-Output_{ds_name}_111.csv', 'w') as output_file:
        r = 'Activation;Explainer;MeanAcc;STDAcc;KappaMean;KappaStd;Time;STDTime\n'
        print(r) 
        output_file.write(r)
        for r in file_text:
            print(r)
            output_file.write(r + '\n')

