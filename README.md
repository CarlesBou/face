# Achieving faithful explainability in feedforward neural networks through accurately computed feature attribution
This repository contains the supplementary material for the paper (pending of publication):

Carles-Bou, Jose L. and Carmona, Enrique J. **Achieving faithful explainability in feedforward neural networks through accurately computed feature attribution**. 

## Paper Abstract
The rapid advancements in machine learning have led to the deployment of complex models in critical domains such as 
healthcare, finance, and autonomous systems. Despite their remarkable predictive performance, the opaque nature of 
these models presents significant challenges for interpretability, which is essential for trust, accountability, and 
regulatory compliance. Explainable Artificial Intelligence (XAI) has emerged as a crucial field addressing these 
challenges by making black-box models more transparent. In this paper, we propose a novel local post-hoc explanation 
method for feedforward neural networks (FNNs) based on a new mathematical model that enables the exact calculation of 
input feature attributions for individual predictions. Our method provides clear and precise explanations while 
achieving 100% fidelity in reflecting the true behavior of the model. Furthermore, our approach demonstrates superior 
performance compared to state-of-the-art XAI techniques, both in terms of explanation fidelity and computational efficiency, 
making it a highly effective and scalable solution. We validate the method through extensive experiments, showcasing its 
versatility across different types of problems. This work enhances interpretability and trust in AI systems by providing a 
reliable explanation framework applicable across a wide range of scenarios modeled with FNNs.


## Repository description
The repository contains the source files developed to implement the new local post-hoc explainer for Feed-forward Neural 
Networks (FNNs) named "**Feature Attribution Computed Exactly (FACE)**" and the code for the different experiments run to 
test the validity of the proposal. 

It has been implemented in Python and uses Keras and Tensorflow framework. 

The *face* folder keeps the source files of the new explainer and the base Keras FNN Model.

The *experiments* contains the diverse experiments made.

The *experiments/datasets* folder contains the datasets employed for classification and regressions tasks used in the experiments.



## 


## Licenses
** PENDIENTE REVISAR LICENCIAS (nuestras y de terceros)
This work is dual-licensed under Creative Commons Zero v1.0 Universal (or any later version) license for our contribution and 
correspondig licenses coming from LIME, SHAP, and FCP authors.
