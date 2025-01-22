# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

from face.fnnmodel import FNNModel
from face.kerasface import face_contrib
from face.kerasface import hard_sigmoid, hard_tanh

import shap
from lime.lime_tabular import LimeTabularExplainer
from fcp.fcp import forward_composition


import keras
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_DETERMINISTIC_OPS"] = '1'

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense 
import tensorflow as tf



import pandas as pd
import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from face_utils import get_lime_contrib, plot_bar_contrib

tf.config.experimental.enable_op_determinism()


       
'''
CLASSIFICATION
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


test_size = 0.20
validation_split = 0.10
epochs = 30        
model_version = 0
use_saved_model_weights = False

n_args = len(sys.argv) 
    
liver_drinks = 7

if n_args == 1:
    ds_name = 'liver'
    # ds_name = 'pima'
    # ds_name = 'breast'
    # ds_name = 'german'
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
    
    
    #y = pd.Categorical(y).codes
    
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


''' 
Creamos y entrenamos el modelo con la definición específica para cada dataset
'''
model = FNNModel(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='nadam',
              metrics=['accuracy'],
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


'''
Empezamos el análisis
'''


print('\nCLASSIFICATION OF INDIVIDUALS\n')
print(f'Using dataset: {dataset_name}, {len(y)} samples ({len(y_train)} train / {len(y_test)} test), {num_inputs} features, {num_outputs} classes')

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


'''
Liver samples
In paper in Oct, 10th, sec 4.3.2 Classification
'''
samples = [43]

y_test_lime = []

save_file = False

for sample in samples:
    
    '''
    CLASSIFICATION: PIRO
    '''
    
    print(f'Plotting PIRO feature contribution for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')
    
    piro_contrib = face_contrib(X_test.iloc[sample], model)
    
    y_piro = np.argmax(np.sum(piro_contrib, axis=1))
    
    '''
    Pintamos las contribuciones ponderadas (coef de contrib * x)
    '''    
    for cl in range(num_outputs):
        plot_bar_contrib(feature_names=feature_names, 
                        contrib_class=piro_contrib[cl],
                        pred_class=y_piro,
                        real_class=y_test[sample],
                        sample_id=sample, show='individual',
                        selected_class=cl, 
                        title=f'Truth/Net/Exp={y_test[sample]}/{y_mlp[sample]}/{y_piro} Intercept={piro_contrib[cl, 0]:.04f}',
                        max_features=num_inputs,
                        mode='classification',
                        reverse_colors=False,
                        normalize=False,
                        show_title=True,
                        save_file=save_file,
                        graph_fname=f'individual_classification_graph_output-{dataset_name}_sample_{sample}_class_{cl}_PIRO.svg',
                        )

    print('OK')
    
   
    '''
    LIME
    '''    
    times_accuracy_mean = 1
    
    print(
        f'Plotting LIME for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')
    
    lime_explainer = LimeTabularExplainer(X_train.to_numpy(),
                                        feature_names=feature_names,
                                        mode='classification',
                                        discretize_continuous=False,
                                        verbose=False,
                                        random_state=33
                                        )
    
    lime_contrib_list = []
    lime_prediction_proba_list = []
    
    for i in range(times_accuracy_mean):
        # y_lime = []
        
        lime_contrib, lime_prediction, lime_prediction_proba = \
            get_lime_contrib(lime_explainer,
                            X_test.iloc[sample].to_numpy(),
                            model.predict_proba,
                            num_inputs, num_outputs)
 
        lime_contrib_list.append(lime_contrib)
        lime_prediction_proba_list.append(lime_prediction_proba)
        
    lime_contrib = np.array(lime_contrib_list).mean(axis=0)
    lime_prediction_proba = np.array(lime_prediction_proba_list).mean(axis=0)
    
    y_lime = np.argmax(lime_prediction_proba)
    
    '''
    Pintamos las contribuciones ponderadas (coef de contrib * x)
    '''
    lime_contrib_weighted = lime_contrib[:,1:] * X_test.iloc[sample].to_numpy()
    
    lime_contrib = np.hstack((lime_contrib[:,0].reshape(-1,1), lime_contrib_weighted)) 
    
    
    print(f'LIME prediction, Class = {lime_prediction} ... ', end='')
    
    for cl in range(num_outputs):
      plot_bar_contrib(feature_names=feature_names, contrib_class=lime_contrib[cl],
                        pred_class=lime_prediction,
                        real_class=y_test[sample],
                        sample_id=sample, show='individual',
                        selected_class=cl,
                        title=f'Truth/Net/Exp={y_test[sample]}/{y_mlp[sample]}/{y_lime} Intercept={lime_contrib[cl,0]:.04f}',
                        max_features=num_inputs,
                        mode='classification',
                        reverse_colors=False,
                        normalize=False,
                        show_title=True,
                        save_file=save_file,
                        graph_fname=f'individual_classification_graph_output-{dataset_name}_sample_{sample}_class_{cl}_LIME.svg',
                        )

    print('OK')
    

    if y_piro != y_lime:
        print(f'\nDetected PIRO and LIME discrepancy in sample {sample}')
        if y_lime != y[sample] and y_piro == y[sample]:
            print('  and PIRO is right')
        print()
        
        assert(0)
        
        
    '''
    SHAP
    '''

    print(
        f'Plotting SHAP for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')


    '''
    Usamos ahora para el background 1 único ejemplo medio del dataset
    '''
                                       
    shap_explainer = shap.KernelExplainer(model.predict_proba, X.mean())
                                          

    shap_explanation = shap_explainer(X_test.iloc[sample])

    shap_values = shap_explanation.values

    shap_contrib = np.hstack((shap_explainer.expected_value.reshape(-1,1), shap_values.T))
    
    y_shap = np.argmax(np.sum(shap_contrib, axis=1))
    
    for cl in range(num_outputs):
      plot_bar_contrib(feature_names=feature_names, contrib_class=shap_contrib[cl],
                        pred_class=y_shap, 
                        real_class=y_test[sample],
                        sample_id=sample, show='individual',
                        selected_class=cl,
                        title=f'Truth/Net/Exp={y_test[sample]}/{y_mlp[sample]}/{y_shap} Expected value={shap_contrib[cl,0]:.04f}',
                        max_features=num_inputs,
                        mode='classification',
                        reverse_colors=False,
                        show_title=True,
                        normalize=False,
                        save_file=save_file,
                        graph_fname=f'individual_classification_graph_output-{dataset_name}_sample_{sample}_class_{cl}_SHAP.svg'
                        )
        
    print('OK')
    

    

    '''
    FCP
    '''

    print(
        f'Plotting FCP for test sample {sample} ground/net={y_test[sample]}/{y_mlp[sample]} ... ', end='')

    fcp_values = forward_composition(model, X_test.iloc[sample])[-1]

    fcp_contrib = np.hstack((np.zeros(num_outputs).reshape(-1,1), fcp_values))

    y_fcp = np.argmax(np.sum(fcp_contrib, axis=1))
    

    for cl in range(num_outputs):
      plot_bar_contrib(feature_names=feature_names, contrib_class=fcp_contrib[cl],
                        pred_class=y_fcp, 
                        real_class=y_test[sample],
                        sample_id=sample, show='individual',
                        selected_class=cl,
                        title=f'Truth/Net/Exp={y_test[sample]}/{y_mlp[sample]}/{y_fcp}',
                        max_features=num_inputs,
                        mode='classification',
                        reverse_colors=False,
                        normalize=False,
                        show_title = True,
                        save_file=save_file,
                        graph_fname=f'individual_classification_graph_output-{dataset_name}_sample_{sample}_class_{cl}_FCP.svg',
                        )
        
    print('OK')
    
    

'''
Pintado de dependencia de parámetros en LIME
'''

# Comentado el 18/ene
# sns.set(font_scale=1.75)
# sns.set_style('whitegrid')

# pima = pd.read_csv("lime_classification_comparison_output-Pima Diabetes_LIME_results.csv", sep=';')
# gpima = sns.catplot(data=pima, x='num-of-samples', hue='$\\alpha_k$',
#                     y='accuracy', legend_out=False, kind='box', height=7, 
#                     dodge=True, palette='pastel')

# plt.savefig('lime_classification_comparison_output-Pima Diabetes_LIME_parameter_dependency.svg', bbox_inches='tight')

# liver = pd.read_csv("lime_classification_comparison_output-Liver disorder_LIME_results.csv", sep=';')
# gliver = sns.catplot(data=liver, x='num-of-samples', hue='$\\alpha_k$', 
#                      y='accuracy', legend_out=False, kind='box', height=7, 
#                      dodge=True, palette='pastel')

# plt.savefig('lime_classification_comparison_output-Liver disorder_LIME_parameter_dependency.svg', bbox_inches='tight')

# sns.set(font_scale=1)


