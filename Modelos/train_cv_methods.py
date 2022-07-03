# Parameter tunning libraries
import optuna
from sklearn.model_selection import GridSearchCV


# sklearn
def train_GridSearchCV(model, param_grid, X_train, X_test, y_train, y_test):
    '''
    Función para realizar el entrenamiento y el ajuste de parámetros con un método de cross-validación de sklearn.
    '''
    grid_search_RF = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, return_train_score=True)
    grid_search_RF.fit(X_train, y_train)
    model_RF_opt = grid_search_RF.best_estimator_
    
    return grid_search_RF.cv_results_

def top_acc_GridSearchCV(results):
    '''
    Devuelve el máximo accuracy alcanzado para un modelo en un conjunto de intentos con GridSearchCV.
    '''
    top_acc = 0
    
    for result in results:
        if result > top_acc:
            top_acc = result
            
    return top_acc

def models_same_acc_GridSearchCV(results, accuracy):
    '''
    Devuelve todas las combinaciones de hiperparámetros con idéntica accuracy en la partición de 
    entrenamiento al entrenar sobre un método de GridSearchCV.
    '''
    models = []
    
    for i, model_acc in enumerate(results["mean_test_score"]):
        if model_acc == accuracy:
            models.append(results["params"][i])
            
    return models


# optuna
def top_acc_OptunaSearchCV(trials):
    '''
    Devuelve el máximo accuracy alcanzado para un modelo en un conjunto de intentos con OptunaSearchCV.
    '''
    top_acc = 0
    
    for trial in trials:
        if trial.value > top_acc:
            top_acc = trial.value
            
    return top_acc

def models_same_acc_OptunaSearchCV(trials, accuracy):
    '''
    Devuelve todas las combinaciones de hiperparámetros con idéntica accuracy en la partición de 
    entrenamiento al entrenar sobre un método de OptunaSearchCV.
    '''
    models = []
    
    for trial in trials:
        if trial.value == accuracy:
            models.append(trial.params)
            
    return models
