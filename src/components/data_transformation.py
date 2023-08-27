import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformationConfig:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifact','proprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course']
            
            numerical_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ])
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encording',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ])
            logging.info('Numerical column encording completed')
            logging.info('categorical column encording completed')

            # combine both the pipeline
            preprocessor = ColumnTransformer(
                [
                    ('Numerical pipeline',numerical_pipline,numerical_features),
                    ('Categorical pipeline',categorical_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info('train and test data completed')
            logging.info('obtain preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = 'math_score'

            #input train
            input_feature_train_df  = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            # input test
            input_feature_test_df   = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df  = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            return  (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            

