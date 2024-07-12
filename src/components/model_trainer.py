import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            model_params ={
                "LinearRegression": {
                    "model": LinearRegression(),
                    'params':{}
                },

                'Lasso':{
                    'model': Lasso(),
                    'params':{
                        'alpha' : [0.1,1,10]
                    }
                },
                
                'Ridge':{
                    'model': Ridge(),
                    'params': {
                        'alpha' : [0.1,1,10]
                    }
                },
                
                'svm':{
                    'model': SVR(gamma='auto'),
                    'params': {
                        'C': [1,10,20],
                        'kernel': ['rbf','linear']
                    }
                },

                'DecisionTree':{
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'criterion' : ["squared_error", "absolute_error"]
                    }
                },

                'RandomForest' : {
                    'model': RandomForestRegressor(),
                    'params' : {
                        'n_estimators' : [1,5,10]
                    }
                },
            }

            logging.info("Model run initiated")
            model_report:dict = evaluate_models(X_train, X_test, y_train, y_test,model_params)
            logging.info("Model run successfully")

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = model_params[best_model_name]["model"]
            logging.info("Best model & parameters extracted")

            if best_model_score < 0.6:
                raise CustomException("No best model found!")
            
            logging.info(f"model report is {model_report}")

            logging.info(f"Best model {best_model_name} & and parameters {best_model.get_params()}")
            
            logging.info("Model saving initiated")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,

                obj = best_model
            )
            logging.info("Model saved successfully")

            predicted = best_model.predict(X_test)

            r2_score_ = r2_score(y_test, predicted)
            logging.info("Model training completed successfully")

            return r2_score_

        except Exception as e:
            raise CustomException(e,sys)

