import mlflow.sklearn
import sklearn.datasets
import sklearn.tree

# 各種パスを指定
TRACKING_URI = '/home/taka1204/for_mlflow_study/tracking_server/mlruns'
EXPERIMENT_NAME = 'test2'
# トラッキングサーバ（バックエンド）の場所を指定
mlflow.set_tracking_uri(TRACKING_URI)
# Experimentの生成
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:  # 当該Experiment存在しないとき、新たに作成
    experiment_id = mlflow.create_experiment(
                            name=EXPERIMENT_NAME)
else: # 当該Experiment存在するとき、IDを取得
    experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    # irisiでモデル生成
    iris = sklearn.datasets.load_iris()
    tree = sklearn.tree.DecisionTreeClassifier()
    model = tree.fit(iris.data, iris.target)

    # metricsらしきものをなんとなく入れておく
    accuracy = sum(model.predict(iris.data) == iris.target) / len(iris.target) 
    mlflow.log_metrics({'accuracy': accuracy})

    # dicision_treeという名前でlog_model
    mlflow.sklearn.log_model(model, 'dicision_tree')