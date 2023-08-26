import os
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestionConfig:

    def __init__(self):
        self.train_data_split = os.path.join('artifact','train.csv')
        self.test_data_split = os.path.join('artifact','test.csv')
        self.raw_data_split = os.path.join('artifact','data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestionconfig = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion method and components')
        try:
            df = pd.read_csv('notebook\datas\stud.csv')
            logging.info('Read data as dataframe')

            os.makedirs(os.path.dirname(self.ingestionconfig.raw_data_split),exist_ok=True)
            # upload the data into this folder
            df.to_csv(self.ingestionconfig.raw_data_split,header=True,index=False)

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
            train_set.to_csv(self.ingestionconfig.train_data_split,index=False,header=True)
            test_set.to_csv(self.ingestionconfig.test_data_split,index=False,header=True)
            logging.info('Ingestion of data is completed')
        except Exception as e:
            raise CustomException(e,sys)
        


            





    