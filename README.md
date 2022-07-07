# Kaggle_MLSP_Schizophrenia_Classification
Code for the participation in the 2014 Kaggle competition: MLSP Schizophrenia Classification Challenge (diagnose of schizophrenia using multimodal features from MRI scans)


## Resumen
El objetivo es llevar a cabo una investigación que busca dar respuesta al interrogante sobre la existencia o no de una base biológica, y por tanto objetiva, para la esquizofrenia en el cerebro. Para ello, se desarrolla un estudio de la búsqueda de un algoritmo de clasificación, que toma de base una serie de medidas (variables numéricas), obtenidas a través del tratamiento mediante diferentes métodos computacionales de imágenes de resonancia magnética de dos clases: sMRI (estructural) y fMRI (funcional).

Este repositorio recoge el código de todas las pruebas realizadas y técnicas implementadas.


## Descripción del código

* ``AnalisisDatos.ipynb``: contiene todas las pruebas relativas a un análisis inicial de los datos. Gráficas de distribución de los mismos, mapas de calor con los valores de correlación de la variable objetivo, búsqueda y tratamiento de outliers, ...
* ``DataAugmentation.ipynb``: implementa 3 formas de introducir ruido en los datos y compara el rendimiento de un modelo de prueba sobre el conjunto de datos original frente al rendimiento sobre los 3 conjuntos aumentados de datos.
* ``Preprocesado.ipynb``: notebook que realiza un estudio sobre la utilidad de las técnicas de pre-procesado de los datos para el problema.
* **Modelos:**
  * ``Anexos.ipynb``: consideraciones sobre algunos algoritmos de optimización.
  * ``Autoencoder.ipynb``: pruebas para el entrenamiento de modelos de autoencoder.
  * ``RandomForest.ipynb``: pruebas para el entrenamiento de modelos de random forest.
  * ``RedesNeuronales - Optimización.ipynb``: pruebas para el entrenamiento de modelos de redes neuronales.
  * ``RedesNeuronales - Otras pruebas.ipynb``: aplicación de técnicas de ponderación de los errores de clasificación y de aprendizaje semi-supervisado para tratar de mejorar los resultados de las redes neuronales.
  * ``XGBoost.ipynb``: pruebas para el entrenamiento de modelos de xgboost.
