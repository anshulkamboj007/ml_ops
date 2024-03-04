import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

exp_id =mlflow.create_experiment('loan prediction')

with mlflow.start_run(run_name='decision tree') as run:
    pass

mlflow.end_run()