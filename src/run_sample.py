# coding: UTF-8

from pprint import pprint

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
import mlflow
from mlflow import log_metric, log_param, log_artifacts

def main():
    # 各種パスを指定
    TRACKING_URI = '/home/taka1204/for_mlflow_study/tracking_server/mlruns'
    EXPERIMENT_NAME = 'test1'
    # トラッキングサーバ（バックエンド）の場所を指定
    mlflow.set_tracking_uri(TRACKING_URI)
    # Experimentの生成
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:  # 当該Experiment存在しないとき、新たに作成
        experiment_id = mlflow.create_experiment(
                                name=EXPERIMENT_NAME)
    else: # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id
    
    iris = load_iris()
    
    gnb = GaussianNB()

    scoring = {"p": "precision_macro",
               "r": "recall_macro",
               "f":"f1_macro"}

    skf = StratifiedKFold(shuffle=True, random_state=0)
    scores = cross_validate(gnb, iris.data, iris.target,
                            cv=skf, scoring=scoring)

    pprint(scores)

    tags = {"learning_time": "3 months",
            "test_time": "3week"}

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_text("text1", 'test test test')
        for i in range(len(scores)):
            log_metric("test_f", scores["test_f"][i])
            log_metric("test_p", scores["test_p"][i])
            log_metric("test_r", scores["test_r"][i])

if __name__ == "__main__":
    main()