import os
from Thyroid_Disease import logger
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from Thyroid_Disease.config.configuration import DataTransformationConfig
import pandas as pd
import numpy as np

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data = pd.read_csv(config.data_path)

    def train_test_split(self) -> None:


        train, test = train_test_split(self.data,test_size=0.2, random_state= 42)

        #saving train test as csv

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)

        logger.info("Splited data into test and train sets")
        logger.info(f"train shape:{train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(train.shape, test.shape)

    def dropColumns(self) -> None:
        name = ['TSH_measured', 'T3_measured', 'TT4_measured', 
                'T4U_measured', 'TBG' , 'FTI_measured','TBG_measured','TSH']
        
        self.data.drop(name, inplace=True, axis=1, errors="ignore")
        logger.info(f"Columns are dropped from DataFrame : {name}")
    
    def replaceWithNull(self) -> None:
        try:
            self.data.replace({'?':np.nan},inplace=True)
            logger.info(" ? is filled with np.nan values")
        except Exception as e:
            logger.info("Error while replacing value with NaN ")
            raise e
        
    def replaceCategorical(self) -> None:
        # converting Sex column as F:0 and M:1
        # rest f:0 and t:1
        self.data["sex"] = self.data["sex"].map({'F':0, 'M':1})

        for col in self.data.columns:
            if len(self.data[col].unique())==2:
                self.data[col] = self.data[col].map({'f':0,'t':1})

        # creating dummies columns of referral source
        self.data = pd.get_dummies(self.data, columns=['referral_source'], dtype='int')
        
        logger.info("replacement of categorical data into Numerical is completed")

    def labelEncoding(self) -> None:

        # making Label Encoding object
        lblEn = LabelEncoder()
        # fitting Label encoding in class column
        self.data["Class"] = lblEn.fit_transform(self.data['Class'])

        logger.info("Label Encoding is completed on Class columns")
        # saving  as  lable_encoding object as pickle

        logger.info("Pickle file of label object is save in location : PATH ")
    
    def fill_na(self) -> None:
        
        # creating imputer object
        imputer = KNNImputer(n_neighbors=5)
        col = self.data.columns
        # applying imputer and getting np.array
        self.data = np.round(imputer.fit_transform(self.data))
        # converting array into dataframe
        self.data = pd.DataFrame(data=self.data, columns=col)
        logger.info(" Empty NaN value is replaced using KNNImputer ")

    def cap_data(self, series):
        # creating lower percentile and upper percentile of outler data
        lower = np.percentile(series ,5)
        upper = np.percentile(series,95)
        return np.clip(series,lower,upper)
    
    def apply_cap(self) -> None:
        # this will apply Cap to Outliers in the following columns
        name = ['age','T3','TT4','T4U','FTI']
        self.data[name] = self.data[name].apply(self.cap_data)
        logger.info(f" Capping outlier completed in columns : {name}")

    def over_sampling(self) -> None:
        
        # smote object
        rdsmple = RandomOverSampler()
        X = self.data.drop(["Class"],axis=1)
        y = self.data["Class"]
        col_name = X.columns

        # Applying Smote and it will return two numpy series
        X, y = rdsmple.fit_resample(X,y)

        # converting oversampled data into dataframe
        self.data = pd.concat([pd.DataFrame(X, columns=col_name),
                               pd.Series(y, name = 'Class')], axis=1)
        logger.info("Data Over Sampling is completed !!!!")



    def run(self) -> None:
        self.dropColumns()
        self.replaceWithNull()
        self.replaceCategorical()
        self.labelEncoding()
        self.fill_na()
        self.apply_cap()
        self.over_sampling()
        self.train_test_split()
