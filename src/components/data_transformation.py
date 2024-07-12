import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try: 
            # Define num and cat columns 
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns_oneHotEnc = ['gender', 'race_ethnicity', 'lunch', 'test_preparation_course']
            categorical_columns_ordEnc = ['parental_level_of_education']
            # Define the categories and their order
            categories = [['some high school', 
                            'high school',
                            'some college',
                            "associate's degree",
                            "bachelor's degree",
                            "master's degree"]]

            # create pipeline for num and cat columns
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline_oneHotEnc = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("oneHotEnc",OneHotEncoder(sparse_output=False,drop="first")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline_ordEnc = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordenc",OrdinalEncoder(categories=categories)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            # Log for num and cat columns
            logging.info(f'categorical columns under going one hot encoding : {categorical_columns_oneHotEnc}')
            logging.info(f'categorical columns under going ordinal encoding : {categorical_columns_ordEnc}')
            logging.info(f'numerical columns under going standard scaling : {numerical_columns}')

            # ColumnTransformer for above pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline_oneHotEnc",cat_pipeline_oneHotEnc,categorical_columns_oneHotEnc),
                    ("cat_pipeline_ordEnc",cat_pipeline_ordEnc,categorical_columns_ordEnc)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test datasets completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # setting target col
            target_column_name = "math_score"

            # getting X and y from train_df
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            # getting X and y from test_df
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Applying preprocessing object to train & test df
            logging.info("Applying preprocessing object to train & test df")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) # fit_transform
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df) # only transform

            # Concatenate input features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Saving preprocessing object
            logging.info("Saving preprocessing object")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Data transformation done!")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

