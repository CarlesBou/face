# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:49:07 2025

@author: Carles
"""

import numpy as np
from matplotlib import pyplot as plt

def plot_bar_contrib(feature_names, contrib_class,
                     pred_class=None, real_class=None, sample_id=None, 
                     show='individual', selected_class=None,
                     title='', max_features=12, 
                     mode='classification',
                     normalize=False, reverse_colors=False,
                     graph_fname=None,
                     show_title=False,
                     save_file=False,
                     bias_name=None):
    
    font_size=35
      
    n_features = min(len(feature_names), max_features)
    
    if normalize: 
      contrib_class = (np.abs(contrib_class) / np.abs(contrib_class).sum()) * np.sign(contrib_class)
      
    bias = contrib_class[0]
    contrib_class = contrib_class[1:]
    
    order = np.argsort(abs(contrib_class))[::-1]
    
    contrib_class = contrib_class[order][:n_features]
    feature_names = feature_names[order][:n_features]
      
    if not reverse_colors:
      color = ['b' if c >= 0 else 'r' for c in contrib_class]
    else:
      color = ['r' if c >= 0 else 'b' for c in contrib_class]
      
    if len(feature_names) < 7:
        fig_height = 7.5
    else:
        fig_height = 9
        
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    
    # Horizontal Bar Plot
    bars = ax.barh(feature_names, abs(contrib_class), color=color, height=0.7)
    
    plt.yticks(fontsize=font_size)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    # ax.xaxis.set_tick_params(pad = 20)
    ax.yaxis.set_tick_params(pad = 10)
    
    for label in ax.get_xticklabels(which='major'):
        label.set(fontsize=font_size - 5)
    
    # Add x, y gridlines
    ax.grid(color ='grey', linestyle ='-.', 
            linewidth = 0.8, alpha = 0.2)
    
    ax.set_ylabel('Feature', fontsize=font_size, labelpad=15)
    ax.set_xlabel('Feature attribution', fontsize=font_size, loc='center', labelpad=15)
   
    ax.invert_yaxis()

    labels = [f'{c:.4f}' for c in contrib_class]
    label_width_points = 0
    
    for bars in ax.containers:
        for i, bar in enumerate(bars):
            y = bar.get_y()
            width = bar.get_width()
            height = bar.get_height()
            
            bar_width_points = bar.get_tightbbox().width
            
            if i == 0:
                text = ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                   xytext=(0, 0), ha='center', va='center', 
                                   fontsize=font_size, color='white',
                                   textcoords="offset points")
                label_width_points = text.get_tightbbox().width
            else:
                if bar_width_points >= label_width_points * 1.10:
                    ax.annotate(labels[i], (width/2, y + height/2 + 0.04), 
                                xytext=(0, 0), ha='center', va='center', 
                                fontsize=font_size, color='white',
                                textcoords="offset points")
                else:
                    ax.annotate(labels[i], (width, y + height/2 + 0.04), 
                                xytext=(6, 0), ha='left', va='center', 
                                fontsize=font_size, color='black',
                                textcoords="offset points")

    
    if show_title:
        # Add Plot Title
        if mode == 'classification':
          if show == 'individual':
                ax.set_title(title, loc='center', fontsize=font_size)
          elif show == 'mean_class':
              ax.set_title(f'{title}for Class {selected_class} - Intercept={bias:.5f}',
                           loc ='center', fontsize=font_size)
        else:
          ax.set_title(f'{title}',
                       loc ='center', fontsize=font_size)
      
    # Save Plot
    if save_file and graph_fname is not None:
        plt.savefig(graph_fname, bbox_inches="tight",)
      
      # Show Plot
    plt.show()
        


    

def get_lime_contrib(lime_explainer, x, predict_f, 
                     num_inputs, num_outputs,  
                     kernel_width=None, num_samples=None, 
                     return_weighted=False, mode='classification'):
    
    if mode == 'classification':
        if num_samples is None:
            lime_explanation = lime_explainer.explain_instance(x, predict_f,
                                                               # num_samples=num_samples,
                                                               num_features=num_inputs,
                                                               top_labels=num_outputs)
        else:
            lime_explanation = lime_explainer.explain_instance(x, predict_f,
                                                               num_samples=num_samples,
                                                               num_features=num_inputs,
                                                               top_labels=num_outputs,
                                                               verbose=True)
    
        lime_contrib = np.zeros((num_outputs, num_inputs + 1))
    
        for o in range(num_outputs):
            for i in range(num_inputs):
                (idx, val) = lime_explanation.as_map()[o][i]
                lime_contrib[o, idx + 1] = val
    
        for o in range(num_outputs):
            lime_contrib[o, 0] = lime_explanation.intercept[o]
    
        lime_prediction = max(lime_explanation.local_preds_,
                              key=lime_explanation.local_preds_.get)
    
        lime_proba = np.zeros(len(lime_explanation.local_preds_))
        for key, val in lime_explanation.local_preds_.items():
            lime_proba[key] = val
    
    
        if return_weighted:  
            lime_contrib[:, 1:] = lime_contrib[:, 1:] * x
    
        return lime_contrib, lime_prediction, lime_proba
    
    else:
        if num_samples is None:
            lime_explanation = lime_explainer.explain_instance(x, predict_f,
                                                               # num_samples=num_samples,
                                                               num_features=num_inputs,
                                                               top_labels=num_outputs)
        else:
            lime_explanation = lime_explainer.explain_instance(x, predict_f,
                                                               num_samples=num_samples,
                                                               num_features=num_inputs,
                                                               top_labels=num_outputs)
    
        lime_contrib = np.zeros((len(lime_explanation.local_exp), num_inputs + 1))
    
        for o in range(len(lime_explanation.local_exp)):
            for i in range(num_inputs):
                (idx, val) = lime_explanation.as_map()[o][i]
                lime_contrib[o, idx + 1] = val
    
        for o in range(len(lime_explanation.local_exp)):
            lime_contrib[o, 0] = lime_explanation.intercept[o]
    
    
        # lime_prediction = lime_contrib[:,1:] @ X_test.iloc[sample] + lime_contrib[:,0]
        lime_prediction = lime_contrib[:,1:] @ x + lime_contrib[:,0]
    
        lime_proba = np.zeros(len(lime_explanation.local_preds_))
        
        # for key, val in lime_explanation.local_preds_.items():
        #     lime_proba[key] = val
    
        if return_weighted:  
            lime_contrib[:, 1:] = lime_contrib[:, 1:] * x
            
        return lime_contrib, lime_prediction, lime_proba
    
    
    
def get_lime_contrib_regression(lime_explainer, x_sample, predict_f, 
                     num_inputs, num_outputs, kernel_width=None, 
                     num_samples=None, return_weighted=False):
    if num_samples is None:
        lime_explanation = lime_explainer.explain_instance(x_sample, predict_f,
                                                           # num_samples=num_samples,
                                                           num_features=num_inputs,
                                                           top_labels=num_outputs)
    else:
        lime_explanation = lime_explainer.explain_instance(x_sample, predict_f,
                                                           num_samples=num_samples,
                                                           num_features=num_inputs,
                                                           top_labels=num_outputs)

    lime_contrib = np.zeros((len(lime_explanation.local_exp), num_inputs + 1))

    for o in range(len(lime_explanation.local_exp)):
        for i in range(num_inputs):
            (idx, val) = lime_explanation.as_map()[o][i]
            lime_contrib[o, idx + 1] = val

    for o in range(len(lime_explanation.local_exp)):
        lime_contrib[o, 0] = lime_explanation.intercept[o]


    # lime_prediction = lime_contrib[:,1:] @ X_test.iloc[sample] + lime_contrib[:,0]
    lime_prediction = lime_contrib[:,1:] @ x_sample + lime_contrib[:,0]

    lime_proba = np.zeros(len(lime_explanation.local_preds_))
    
    # for key, val in lime_explanation.local_preds_.items():
    #     lime_proba[key] = val

    if return_weighted:  
        lime_contrib[:, 1:] = lime_contrib[:, 1:] * x_sample
        
    return lime_contrib, lime_prediction, lime_proba

