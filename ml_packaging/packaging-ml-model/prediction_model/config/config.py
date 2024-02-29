import os
import pathlib
import prediction_model

PACKAGE_ROOT=pathlib.PATH(prediction_model.__file__).resolve().parent

DATAPATH=os.path.join(PACKAGE_ROOT,"datasets")

TRAIN_FILE='train.csv'
TEST_FILE='test.csv'

SAVE_MODEL_PATH=os.path.join(PACKAGE_ROOT,"trained_models")

MODEL_NAME='classification.pkl'

TARGET='Loan_Status'

#final features used in model
FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount','CoapplicantIncome',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount','Loan_Amount_Term']

CAT_FEATURES =['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area']

#in our case it is same as Categorial features

FEATURES_TO_ENCODE =['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Credit_History', 'Property_Area']

FEATURES_TO_MODIFY ='ApplicantIncome'
FEATURES_TO_ADD ='CoapplicantIncome'

DROP_FEATURES=['CoapplicantIncome']

LOG_FEATURES= ['ApplicantIncome', 'LoanAmount']
