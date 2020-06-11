import joblib
from Preprocess import Preprocess
import ModelConfig
import pandas as pd
import os
from api_logger import logger

class Train(Preprocess):

    def __init__(self, filename):

        self.filename = filename
        super().__init__()
        self.model = joblib.load(open(ModelConfig.model_path, 'rb'))
        self.feature_transform = joblib.load(open(ModelConfig.feature_transform_path, 'rb'))
        self.predict_data = None
        self.id_ = self.input['sku']
        self.prediction = None
        self.output = None


    def imputation(self):
        for col, val in self.training_impute.items():
            self.input[col] = self.input[col].fillna(val)

    def encode_categorical(self, cat_cols=None):
        if cat_cols:
            self.cat_cols = cat_cols

        for col in self.cat_cols:
            self.input[col] = self.input[col].map({'yes': 1, 'no': 0})

    def process_data(self):
        logger.info(f'\n Data Quality Check Starts... \n')
        super().data_quality_check()
        logger.info(f'\n Data Imputation Starts... \n')
        self.imputation()
        logger.info(f'\n Encoding Categorical Starts... \n')
        self.encode_categorical(cat_cols=None)
        self.input.to_csv(os.path.join('../transformation', 'transformed_' + self.filename), index=False)
        logger.info(f"\n Transformed file saved at {os.path.join('../transformation', 'transformed_' + self.filename)}... \n")
        self.predict_data = self.feature_transform.transform(self.input[self.model_features])

    def predict(self):
        self.prediction = self.model.predict(self.predict_data)
        self.output = pd.DataFrame(list(zip(self.id_, self.prediction)),
                                   columns=['sku', 'prediction'])
        self.output["prediction"] = self.output["prediction"].map({0: "Yes", 1: "No"})
        self.output.to_csv(os.path.join('../prediction', 'output_' + self.filename),index = False)
        logger.info(f"\n Predicted file saved at {os.path.join('../prediction', 'output_' + self.filename)}... \n")