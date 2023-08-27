import os
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

class DataIngestionConfig:

    def __init__(self):
        self.train_data_path = os.path.join('artifact','train.csv')
        self.test_data_path = os.path.join('artifact','test.csv')
        self.raw_data_path = os.path.join('artifact','data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestionconfig = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion method and components')
        try:
            df = pd.read_csv('notebook\datas\stud.csv')
            logging.info('Read data as dataframe')

            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path),exist_ok=True)
            # upload the data into this folder
            df.to_csv(self.ingestionconfig.raw_data_path,header=True,index=False)

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
            train_set.to_csv(self.ingestionconfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')

            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

        


            





    