# -*- coding: utf-8 -*-

import mlflow

mlflow.start_run()
mlflow.set_experiment("w")

mlflow.log_param(key='foo', value='bar')
mlflow.log_metric(key='foo', value=2.0)
mlflow.set_tag(key='fruit', value='apple')

mlflow.end_run()
