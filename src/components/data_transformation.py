import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_colms = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']
            numerical_colms = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 'multiple_deliveries', 'Ordered_Time_Hour', 'Ordered_Time_Minute','distance']
            
            # Define the custom ranking for each ordinal variable
            Weather_categories = ["Sunny", "Cloudy", "Sandstorms","Windy", "Stormy", "Fog" ]
            Traffic_categories  = ["Low", "Medium", "High", "Jam"]
            Vehicle_categories = ["electric_scooter", "motorcycle", "scooter","bicycle"]
            festival_categories = ["No", "Yes"]
            City_categories = ["Metropolitian", "Urban", "Semi-Urban"]

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Weather_categories, Traffic_categories, Vehicle_categories, festival_categories, City_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline, numerical_colms),
            ('cat_pipeline',cat_pipeline, categorical_colms)
            ])

            return preprocessor    # will retrun the pickle file preprocessor.

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        

    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)   ## like X_train
            target_feature_train_df=train_df[target_column_name]      ## like y_train

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)     ## like X_test
            target_feature_test_df=test_df[target_column_name]       ## like y_test
                
            ## Transformation using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)   # for train data
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)   ## for test data

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            ## concateanting it and converting into numpy array beacuse it is operated faster.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  ## numpy.c_ returns a concatenated array.
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
                
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)