import pandas as pd
import mlflow
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from pycaret.classification import *
from imblearn.over_sampling import SMOTE

def conform_data_2PT(data):
    data_2p = data[data['shot_type']=="2PT Field Goal"]
    data_2p = data_2p[data_2p['shot_made_flag'].isna()==False]
    df = data_2p[['lat','period','minutes_remaining','lon','playoffs','shot_distance','shot_made_flag']]
    return df

def conform_data_3PT(data):
    data_2p = data[data['shot_type']=="3PT Field Goal"]
    data_2p = data_2p[data_2p['shot_made_flag'].isna()==False]
    df = data_2p[['lat','period','minutes_remaining','lon','playoffs','shot_distance','shot_made_flag']]
    return df

#Corrigindo desbalanceamento das amostras no dataset 3PT Field Goal
def train_data_3PT(data):
    x = data.drop(columns = ['shot_made_flag']).copy()
    y = data['shot_made_flag']

    x_resample,y_resample = SMOTE().fit_resample(x,y)

    x_train,x_test,y_train,y_test = train_test_split(x_resample,y_resample,test_size=0.2,random_state= 42)
    
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return [x_train,x_test,y_train,y_test]

def train_data_2PT(data):
    x = data.drop(columns = ['shot_made_flag']).copy()
    y = data['shot_made_flag']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 42)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return [x_train,x_test,y_train,y_test]

def normalize(x_train):
    df_resumo_estatistico = x_train.describe()
    x_train_norm = (x_train-df_resumo_estatistico.loc['mean'])/df_resumo_estatistico.loc['std']
    
    return x_train_norm 

def train_logistic_regression(x_train,y_train):
    expertiment_name = 'Logistic_Regression'
    experiment = mlflow.get_experiment_by_name(expertiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id, nested= True):
        n_folds = 10
        params = {
                'penalty':['l1','l2'],
                'C':[1,2,10]
                }
        cv =StratifiedKFold(n_splits = n_folds)
        model_template =  LogisticRegression(solver = 'saga', max_iter = 10000)
        clf = GridSearchCV(
                model_template,
                params,
                cv = cv,
                scoring=['f1','precision','recall'],
                refit = 'f1',
                return_train_score=True,
                n_jobs = 4
                )
        clf.fit(x_train,y_train)
        report_model(clf,params,n_folds)
        return clf

def best_classification (data):
    experiment_name = "best_classificator"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id = experiment.experiment_id, nested = True):
    
        n_folds = 10
        params = {
                'penalty':['l1','l2'],
                'C':[1,2,10]
                }

        cv =StratifiedKFold(n_splits = n_folds)

        s = setup(data, target = 'shot_made_flag', session_id= 123)
        
        compare_models()
        lr = create_model('lr')

        tuned_lr = tune_model(lr)
        predictions = predict_model(tuned_lr)

        clf = save_model(tuned_lr,'best_lr')

        predictions 
        mlflow.log_metric('Acurácia',pull()["Accuracy"])
        mlflow.log_metric('AUC',pull()["AUC"])
        mlflow.log_metric('F1',pull()["F1"])
        mlflow.log_metric('Recall',pull()["Recall"])
        return clf

def report_model(clf,params,n_folds):

    idx = clf.best_index_

    mlflow.log_metric('f1_std',clf.cv_results_['std_test_f1'][idx])
    mlflow.log_metric('f1_mean',clf.cv_results_['mean_test_f1'][idx])

    mlflow.log_metric('precision_std',clf.cv_results_['std_test_precision'][idx])
    mlflow.log_metric('precision_mean',clf.cv_results_['mean_test_precision'][idx])
 
    mlflow.log_metric('recall_std',clf.cv_results_['std_test_recall'][idx])
    mlflow.log_metric('recall_mean',clf.cv_results_['mean_test_recall'][idx])  

    mlflow.log_param('model','logistic_regression')

    mlflow.log_param('parameters',params)
    mlflow.log_param('n_folds',n_folds)

    mlflow.sklearn.log_model(clf, "model")


