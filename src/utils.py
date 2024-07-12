'''
This module is responsible for methods common to all file
'''

import os
import sys
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, X_test, y_train, y_test,model_params):
    try:
        report = {}
        for model_name, mp in model_params.items():
            # Perform GridSearchCV
            clf = GridSearchCV(mp['model'],mp['params'], cv =5, return_train_score = False)
            clf.fit(X_train,y_train)

            # Set best params to that model
            mp['model'].set_params(**clf.best_params_)
            mp['model'].fit(X_train,y_train)

            y_train_pred = mp['model'].predict(X_train)

            y_test_pred = mp['model'].predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)

