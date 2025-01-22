# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 05:28:18 2024

@author: Carles
"""

import pandas as pd
from tensorflow.keras.models import Model
from scipy.special import softmax
from sklearn.metrics import accuracy_score


class FNNModel(Model):
    def predict(self, *args, **kwargs):
      xx = args[0]
      if len(args[0].shape) == 1:
        if isinstance(args[0], (pd.DataFrame, pd.Series)):
          xx = args[0].to_numpy().reshape(1, -1)
        else:
          xx = args[0].reshape(1, -1)
        
      kwargs['verbose'] = False
      
      return super().predict(xx, *args[1:], **kwargs)

    def predict_proba(self, *args, **kwargs):
      kwargs['verbose'] = False  
      ret = self.predict(*args, **kwargs)
      return softmax(ret, axis=1)

    def score(self, y, y_pred):
        return accuracy_score(y, y_pred)

