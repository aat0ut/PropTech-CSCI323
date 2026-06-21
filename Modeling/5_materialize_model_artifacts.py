import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def parse_args():
    root=Path(__file__).resolve().parents[1]
    modeling_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-path",type=Path,default=root/"Data/Clean Data/train.csv")
    parser.add_argument("--model-dir",type=Path,default=modeling_dir/"models")
    parser.add_argument("--result-dir",type=Path,default=modeling_dir/"results")
    parser.add_argument("--train-limit",type=int,default=0)
    parser.add_argument("--meta-fraction",type=float,default=0.15)
    parser.add_argument("--random-state",type=int,default=42)
    parser.add_argument("--xgb-estimators",type=int,default=500)
    parser.add_argument("--xgb-depth",type=int,default=10)
    parser.add_argument("--xgb-learning-rate",type=float,default=0.03)
    parser.add_argument("--lgbm-estimators",type=int,default=500)
    parser.add_argument("--lgbm-leaves",type=int,default=127)
    parser.add_argument("--rf-estimators",type=int,default=300)
    parser.add_argument("--rf-depth",type=int,default=30)
    return parser.parse_args()


def load_train(path,limit):
    df=pd.read_csv(path,nrows=limit if limit and limit>0 else None)
    y=df["trans_value"]
    X=df.drop(columns=["trans_value","transaction_num"],errors="ignore")
    X.columns=[c.replace(" ","_").replace("-","_") for c in X.columns]
    X=X.astype(float)
    return df,X,y


def build_models(args):
    xgb=XGBRegressor(
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=-1,
        subsample=0.8,
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_depth,
        learning_rate=args.xgb_learning_rate,
        colsample_bytree=0.8
    )
    lgbm=LGBMRegressor(
        random_state=args.random_state,
        verbose=-1,
        num_leaves=args.lgbm_leaves,
        n_estimators=args.lgbm_estimators,
        max_depth=-1,
        learning_rate=0.05
    )
    rf=RandomForestRegressor(
        random_state=args.random_state,
        n_jobs=-1,
        n_estimators=args.rf_estimators,
        min_samples_split=2,
        min_samples_leaf=2,
        max_depth=args.rf_depth
    )
    return {"xgb":xgb,"lgbm":lgbm,"rf":rf}


def metrics(y_true,pred):
    return {
        "MAPE":float(mean_absolute_percentage_error(y_true,pred)),
        "RMSE":float(np.sqrt(mean_squared_error(y_true,pred))),
        "R2":float(r2_score(y_true,pred))
    }


def main():
    args=parse_args()
    args.model_dir.mkdir(parents=True,exist_ok=True)
    args.result_dir.mkdir(parents=True,exist_ok=True)
    start=time.time()
    print("loading training data",flush=True)
    df,X,y=load_train(args.train_path,args.train_limit)
    split=max(1,min(len(X)-1,int(len(X)*(1-args.meta_fraction))))
    X_base,y_base=X.iloc[:split],y.iloc[:split]
    X_meta,y_meta=X.iloc[split:],y.iloc[split:]
    print(json.dumps({"rows":len(X),"features":X.shape[1],"base_rows":len(X_base),"meta_rows":len(X_meta)},indent=2),flush=True)
    base_models=build_models(args)
    meta_predictions=[]
    base_metrics={}
    for name,model in base_models.items():
        print(f"training {name} meta model",flush=True)
        model.fit(X_base,y_base)
        pred=model.predict(X_meta)
        meta_predictions.append(pred)
        base_metrics[name]=metrics(y_meta,pred)
        print(json.dumps({"model":name,"elapsed_seconds":round(time.time()-start,2),**base_metrics[name]},indent=2),flush=True)
    meta_X=np.column_stack(meta_predictions)
    scaler=StandardScaler()
    meta_X_scaled=scaler.fit_transform(meta_X)
    ridge=RidgeCV(alphas=[0.01,0.1,1.0,10.0,100.0])
    ridge.fit(meta_X_scaled,y_meta)
    stacked_pred=ridge.predict(meta_X_scaled)
    stacked_metrics=metrics(y_meta,stacked_pred)
    print(json.dumps({"model":"stacked","alpha":float(ridge.alpha_),"elapsed_seconds":round(time.time()-start,2),**stacked_metrics},indent=2),flush=True)
    final_models=build_models(args)
    for name,model in final_models.items():
        print(f"training final {name}",flush=True)
        model.fit(X,y)
        joblib.dump(model,args.model_dir/f"{name}.pkl")
        print(json.dumps({"saved":str(args.model_dir/f"{name}.pkl"),"elapsed_seconds":round(time.time()-start,2)},indent=2),flush=True)
    joblib.dump(ridge,args.model_dir/"ridge.pkl")
    joblib.dump(scaler,args.model_dir/"meta_scaler.pkl")
    with open(args.model_dir/"feature_columns.json","w") as f:
        json.dump(list(X.columns),f,indent=2)
    summary={
        "train_path":str(args.train_path),
        "train_rows":int(len(X)),
        "feature_count":int(X.shape[1]),
        "meta_fraction":float(args.meta_fraction),
        "base_rows":int(len(X_base)),
        "meta_rows":int(len(X_meta)),
        "base_metrics":base_metrics,
        "stacked_metrics":stacked_metrics,
        "ridge_alpha":float(ridge.alpha_),
        "ridge_coefficients":ridge.coef_.astype(float).tolist(),
        "elapsed_seconds":round(time.time()-start,2)
    }
    with open(args.result_dir/"artifact_training_summary.json","w") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
