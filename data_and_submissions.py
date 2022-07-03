import pandas as pd
from sklearn.model_selection import train_test_split

import pathlib
from datetime import datetime

def load_data():   
    # Datos de entrenamiento
    trainFNC = pd.read_csv("data/train_FNC.csv")
    trainSBM = pd.read_csv("data/train_SBM.csv")
    train_labels = pd.read_csv("data/train_labels.csv")

    # DataFrame con ambas fuentes de datos
    train = pd.merge(left=trainFNC, right=trainSBM, left_on="Id", right_on="Id")
    data = pd.merge(left=train_labels, right=train, left_on="Id", right_on="Id")
    data.drop("Id", inplace=True, axis=1)

    # Shuffle de los datos de train
    data = data.sample(frac=1, random_state=0)

    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Datos de test
    testFNC = pd.read_csv("data/test_FNC.csv")
    testSBM = pd.read_csv("data/test_SBM.csv")

    # DataFrame con ambas fuentes de datos
    test_kaggle = pd.merge(left=testFNC, right=testSBM, left_on="Id", right_on="Id")
    test_kaggle.drop("Id", inplace=True, axis=1)
    
    return X_train, X_test, y_train, y_test, test_kaggle


def create_submission(pred, alias, current_folder=False):
    '''
    Función para generar un csv con las predicciones de un modelo para participar en la competición de Kaggle
    '''
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = np.argmax(pred, axis=1)

    submission_path = pathlib.Path().resolve()
    if current_folder:
        path = ""
        testFNC = pd.read_csv(f"{path}data/test_FNC.csv")
    else:
        path = "../"
        testFNC = pd.read_csv(f"{path}data/test_FNC.csv")
        submission_path = submission_path.parent
        
    test_id=testFNC["Id"]
    submissionDF = pd.DataFrame(list(zip(test_id, pred)), columns=["Id", "Probability"])
    print(submissionDF.shape) # Comprobación del tamaño, debe ser: (119748, 2)
    current_time = datetime.now().strftime("%d-%m-%Y_%Hh%Mmin")
    
    submissionDF.to_csv(f"{submission_path}\submissions\MLSP_submission_{alias}_{current_time}.csv", header=True, index=False)