import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL_NAMES=("xgb","lgbm","rf")


def parse_args():
    root=Path(__file__).resolve().parents[1]
    modeling_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-path",type=Path,default=root/"Data/Clean Data/test.csv")
    parser.add_argument("--background-path",type=Path,default=root/"Data/Clean Data/train.csv")
    parser.add_argument("--model-dir",type=Path,default=modeling_dir/"models")
    parser.add_argument("--output-dir",type=Path,default=modeling_dir/"results/xai")
    parser.add_argument("--sample-size",type=int,default=500)
    parser.add_argument("--background-size",type=int,default=100)
    parser.add_argument("--selection",choices=("head","random","largest_error","highest_uncertainty","all"),default="head")
    parser.add_argument("--plot-count",type=int,default=10)
    parser.add_argument("--top-features",type=int,default=15)
    parser.add_argument("--random-state",type=int,default=42)
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def require_files(files):
    missing={name:path for name,path in files.items() if not path.exists()}
    if missing:
        lines="\n".join(f"{name}: {path}" for name,path in missing.items())
        raise FileNotFoundError(f"Missing required artifacts:\n{lines}")


def normalize_columns(columns):
    return [c.replace(" ","_").replace("-","_") for c in columns]


def load_matrix(path,feature_columns):
    df=pd.read_csv(path)
    y=df["trans_value"] if "trans_value" in df.columns else None
    X=df.drop(columns=["trans_value","transaction_num"],errors="ignore")
    X.columns=normalize_columns(X.columns)
    missing=[c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")
    X=X[feature_columns].astype(float)
    return df,X,y


def load_stack(model_dir):
    files={
        "xgb":model_dir/"xgb.pkl",
        "lgbm":model_dir/"lgbm.pkl",
        "rf":model_dir/"rf.pkl",
        "ridge":model_dir/"ridge.pkl",
        "scaler":model_dir/"meta_scaler.pkl",
        "features":model_dir/"feature_columns.json"
    }
    require_files(files)
    models={name:joblib.load(files[name]) for name in MODEL_NAMES}
    ridge=joblib.load(files["ridge"])
    scaler=joblib.load(files["scaler"])
    feature_columns=load_json(files["features"])
    return models,ridge,scaler,feature_columns,files


def predict_stack(models,ridge,scaler,X):
    base=np.column_stack([models[name].predict(X) for name in MODEL_NAMES])
    final=ridge.predict(scaler.transform(base))
    spread=base.std(axis=1)
    return base,final,spread


def select_positions(y,predictions,spread,sample_size,selection,random_state):
    n=len(predictions)
    if selection=="all" or sample_size<=0 or sample_size>=n:
        return np.arange(n)
    if selection=="head":
        return np.arange(sample_size)
    if selection=="random":
        rng=np.random.default_rng(random_state)
        return np.sort(rng.choice(n,size=sample_size,replace=False))
    if selection=="largest_error":
        if y is None:
            raise ValueError("largest_error selection requires trans_value")
        errors=np.abs(y.to_numpy(dtype=float)-predictions)
        return np.argsort(errors)[-sample_size:][::-1]
    return np.argsort(spread)[-sample_size:][::-1]


def load_background(background_path,data_path,feature_columns,background_size,random_state):
    path=background_path if background_path.exists() else data_path
    _,X,_=load_matrix(path,feature_columns)
    if background_size>0 and len(X)>background_size:
        X=X.sample(n=background_size,random_state=random_state).sort_index()
    return X


def scalar_value(value):
    array=np.asarray(value,dtype=float)
    return float(array.reshape(-1)[0])


def normalize_shap_values(values):
    if isinstance(values,list):
        if len(values)!=1:
            raise ValueError("Expected one SHAP output for regression")
        values=values[0]
    values=np.asarray(values,dtype=float)
    if values.ndim==3 and values.shape[-1]==1:
        values=values[:,:,0]
    if values.ndim!=2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def tree_shap(model,X,background):
    if model.__class__.__name__=="RandomForestRegressor":
        explainer=shap.TreeExplainer(model)
    else:
        try:
            explainer=shap.TreeExplainer(model,data=background,feature_perturbation="interventional")
        except Exception:
            explainer=shap.TreeExplainer(model)
    try:
        if model.__class__.__name__=="RandomForestRegressor":
            values=explainer.shap_values(X,check_additivity=False,approximate=True)
        else:
            values=explainer.shap_values(X,check_additivity=False)
    except TypeError:
        values=explainer.shap_values(X)
    return normalize_shap_values(values),scalar_value(explainer.expected_value)


def combine_stack_shap(base_shap,base_expected,ridge,scaler):
    coefficients=np.asarray(ridge.coef_,dtype=float).reshape(-1)
    scale=np.asarray(scaler.scale_,dtype=float).reshape(-1)
    mean=np.asarray(scaler.mean_,dtype=float).reshape(-1)
    weights=coefficients/scale
    values=np.zeros_like(next(iter(base_shap.values())))
    expected=np.asarray([base_expected[name] for name in MODEL_NAMES],dtype=float)
    for position,name in enumerate(MODEL_NAMES):
        values+=weights[position]*base_shap[name]
    base_value=float(ridge.intercept_+np.sum(coefficients*((expected-mean)/scale)))
    return values,base_value,weights


def json_safe(value):
    if value is None:
        return None
    if isinstance(value,(np.generic,)):
        value=value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def safe_id(value):
    text="row" if value is None else str(value)
    text=re.sub(r"[^A-Za-z0-9_.-]+","_",text).strip("_")
    return text[:80] or "row"


def build_waterfall_outputs(raw_sample,X_sample,y_sample,predictions,positions,values,base_value,output_dir):
    rows=[]
    long_rows=[]
    feature_names=list(X_sample.columns)
    transaction_values=raw_sample["transaction_num"].tolist() if "transaction_num" in raw_sample.columns else [None]*len(X_sample)
    actual_values=y_sample.to_numpy(dtype=float) if y_sample is not None else [None]*len(X_sample)
    for local_index,source_position in enumerate(positions):
        order=np.argsort(-np.abs(values[local_index]))
        running=float(base_value)
        steps=[]
        for rank,feature_index in enumerate(order,start=1):
            feature=feature_names[feature_index]
            shap_value=float(values[local_index,feature_index])
            before=running
            running+=shap_value
            step={
                "rank":rank,
                "feature":feature,
                "feature_value":json_safe(X_sample.iloc[local_index,feature_index]),
                "shap_value":shap_value,
                "abs_shap_value":abs(shap_value),
                "running_value_before":before,
                "running_value_after":running
            }
            steps.append(step)
            long_rows.append({
                "source_position":int(source_position),
                "local_index":int(local_index),
                "transaction_num":json_safe(transaction_values[local_index]),
                "predicted_value":float(predictions[local_index]),
                "actual_value":json_safe(actual_values[local_index]),
                **step
            })
        rows.append({
            "source_position":int(source_position),
            "local_index":int(local_index),
            "transaction_num":json_safe(transaction_values[local_index]),
            "actual_value":json_safe(actual_values[local_index]),
            "predicted_value":float(predictions[local_index]),
            "base_value":float(base_value),
            "waterfall":steps
        })
    payload={"base_value":float(base_value),"feature_count":len(feature_names),"row_count":len(rows),"rows":rows}
    with open(output_dir/"waterfall_arrays.json","w") as f:
        json.dump(payload,f,indent=2)
    pd.DataFrame(long_rows).to_csv(output_dir/"waterfall_arrays.csv",index=False)
    return payload


def save_feature_importance(values,feature_columns,output_dir):
    importance=pd.DataFrame({
        "feature":feature_columns,
        "mean_abs_shap":np.abs(values).mean(axis=0),
        "mean_shap":values.mean(axis=0)
    })
    importance=importance.sort_values("mean_abs_shap",ascending=False).reset_index(drop=True)
    importance.insert(0,"rank",np.arange(1,len(importance)+1))
    importance.to_csv(output_dir/"feature_importance.csv",index=False)
    return importance


def save_plots(values,base_value,X_sample,output_dir,plot_count,top_features):
    explanation=shap.Explanation(values=values,base_values=np.full(len(X_sample),base_value),data=X_sample.to_numpy(),feature_names=list(X_sample.columns))
    plot_paths=[]
    shap.plots.bar(explanation,max_display=top_features,show=False)
    plt.tight_layout()
    bar_path=output_dir/"summary_bar.png"
    plt.savefig(bar_path,dpi=180,bbox_inches="tight")
    plt.close()
    plot_paths.append(bar_path)
    shap.plots.beeswarm(explanation,max_display=top_features,show=False)
    plt.tight_layout()
    beeswarm_path=output_dir/"summary_beeswarm.png"
    plt.savefig(beeswarm_path,dpi=180,bbox_inches="tight")
    plt.close()
    plot_paths.append(beeswarm_path)
    waterfall_dir=output_dir/"waterfall_assets"
    waterfall_dir.mkdir(parents=True,exist_ok=True)
    transaction_values=None
    if "transaction_num" in X_sample.columns:
        transaction_values=X_sample["transaction_num"].tolist()
    for local_index in range(min(plot_count,len(X_sample))):
        row_explanation=shap.Explanation(values=values[local_index],base_values=base_value,data=X_sample.iloc[local_index].to_numpy(),feature_names=list(X_sample.columns))
        shap.plots.waterfall(row_explanation,max_display=top_features,show=False)
        plt.tight_layout()
        name=safe_id(transaction_values[local_index] if transaction_values else local_index)
        path=waterfall_dir/f"waterfall_{local_index:04d}_{name}.png"
        plt.savefig(path,dpi=180,bbox_inches="tight")
        plt.close()
        plot_paths.append(path)
    return plot_paths


def save_explained_rows(raw_sample,predictions,base_spread,output_dir):
    df=raw_sample.copy()
    df.insert(0,"source_position",raw_sample.index.to_numpy())
    df["xai_predicted_value"]=predictions
    df["xai_base_spread"]=base_spread
    df.to_csv(output_dir/"explained_rows.csv",index=False)


def main():
    args=parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    models,ridge,scaler,feature_columns,artifact_files=load_stack(args.model_dir)
    raw,X,y=load_matrix(args.data_path,feature_columns)
    base_predictions,final_predictions,spread=predict_stack(models,ridge,scaler,X)
    positions=select_positions(y,final_predictions,spread,args.sample_size,args.selection,args.random_state)
    X_sample=X.iloc[positions].copy()
    raw_sample=raw.iloc[positions].copy()
    y_sample=y.iloc[positions] if y is not None else None
    background=load_background(args.background_path,args.data_path,feature_columns,args.background_size,args.random_state)
    base_shap={}
    base_expected={}
    for name in MODEL_NAMES:
        print(f"explaining {name}",flush=True)
        base_shap[name],base_expected[name]=tree_shap(models[name],X_sample,background)
    shap_values,base_value,meta_weights=combine_stack_shap(base_shap,base_expected,ridge,scaler)
    selected_base_predictions=base_predictions[positions]
    selected_predictions=final_predictions[positions]
    selected_spread=spread[positions]
    reconstruction=base_value+shap_values.sum(axis=1)
    np.save(args.output_dir/"shap_values.npy",shap_values)
    np.save(args.output_dir/"base_values.npy",np.full(len(X_sample),base_value))
    np.save(args.output_dir/"base_model_predictions.npy",selected_base_predictions)
    pd.DataFrame(shap_values,columns=feature_columns).to_csv(args.output_dir/"shap_values.csv",index=False)
    save_explained_rows(raw_sample,selected_predictions,selected_spread,args.output_dir)
    importance=save_feature_importance(shap_values,feature_columns,args.output_dir)
    build_waterfall_outputs(raw_sample,X_sample,y_sample,selected_predictions,positions,shap_values,base_value,args.output_dir)
    plot_paths=save_plots(shap_values,base_value,X_sample,args.output_dir,args.plot_count,args.top_features)
    manifest={
        "data_path":str(args.data_path),
        "background_path":str(args.background_path if args.background_path.exists() else args.data_path),
        "model_dir":str(args.model_dir),
        "output_dir":str(args.output_dir),
        "row_count":int(len(X_sample)),
        "feature_count":int(len(feature_columns)),
        "selection":args.selection,
        "sample_size":int(args.sample_size),
        "background_size":int(len(background)),
        "base_value":float(base_value),
        "base_model_expected_values":{name:float(base_expected[name]) for name in MODEL_NAMES},
        "meta_weights":{name:float(meta_weights[index]) for index,name in enumerate(MODEL_NAMES)},
        "shap_modes":{"xgb":"tree_interventional","lgbm":"tree_interventional","rf":"tree_path_dependent_approximate"},
        "reconstruction_max_abs_error":float(np.max(np.abs(reconstruction-selected_predictions))),
        "reconstruction_mean_abs_error":float(np.mean(np.abs(reconstruction-selected_predictions))),
        "top_features":importance.head(args.top_features).to_dict(orient="records"),
        "artifacts":{
            "shap_values_npy":str(args.output_dir/"shap_values.npy"),
            "shap_values_csv":str(args.output_dir/"shap_values.csv"),
            "base_values_npy":str(args.output_dir/"base_values.npy"),
            "base_model_predictions_npy":str(args.output_dir/"base_model_predictions.npy"),
            "feature_importance_csv":str(args.output_dir/"feature_importance.csv"),
            "waterfall_arrays_json":str(args.output_dir/"waterfall_arrays.json"),
            "waterfall_arrays_csv":str(args.output_dir/"waterfall_arrays.csv"),
            "explained_rows_csv":str(args.output_dir/"explained_rows.csv"),
            "plots":[str(path) for path in plot_paths]
        },
        "source_artifacts":{name:str(path) for name,path in artifact_files.items()}
    }
    with open(args.output_dir/"manifest.json","w") as f:
        json.dump(manifest,f,indent=2)
    print(json.dumps({"output_dir":str(args.output_dir),"row_count":len(X_sample),"feature_count":len(feature_columns),"reconstruction_max_abs_error":manifest["reconstruction_max_abs_error"]},indent=2))


if __name__=="__main__":
    main()
