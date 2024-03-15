import mlflow
logged_model = 'runs:/7a138374c8624bbba297766bcc974680/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data=[[1,0,0,0,0,4.98745,360,1,2,8.698]]
print(loaded_model.predict(pd.DataFrame(data)))