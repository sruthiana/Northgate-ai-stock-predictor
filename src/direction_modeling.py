"""Northgate next-day direction classification. Run: python -m src.direction_modeling""" 
import logging, os, warnings
import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from src.config import load_config, project_path

TARGET="Target_Next_Day_Log_Return"
LABELS=["Down","Neutral","Up"]
EXCLUDED={"Date","Ticker",TARGET}
LEAK=("target","future","next_day","forward_return","lead_","label","actual_return","predicted","prediction","signal","recommendation")

def labels(y, band):
    y=np.asarray(y,dtype=float); z=np.full(len(y),"Neutral",dtype=object)
    z[y < -band]="Down"; z[y > band]="Up"; return z

def metric(y,p):
    return {"Observations":len(y),"Accuracy":accuracy_score(y,p),
    "Balanced_Accuracy":balanced_accuracy_score(y,p),
    "Macro_F1":f1_score(y,p,labels=LABELS,average="macro",zero_division=0),
    "Macro_Precision":precision_score(y,p,labels=LABELS,average="macro",zero_division=0),
    "Macro_Recall":recall_score(y,p,labels=LABELS,average="macro",zero_division=0)}

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",force=True)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT",str(os.cpu_count() or 2))
    cfg=load_config(); processed=project_path(cfg["paths"]["processed_data"])
    frames={}
    for s in ("train","validation","test"):
        f=processed/f"{s}.parquet"
        if not f.exists(): raise FileNotFoundError(f"Missing {f}; run modeling_data first.")
        d=pd.read_parquet(f)
        if not {"Date","Ticker",TARGET}.issubset(d.columns): raise ValueError(f"{s} missing Date/Ticker/{TARGET}")
        d["Date"]=pd.to_datetime(d["Date"]); d["Ticker"]=d["Ticker"].astype(str)
        frames[s]=d.sort_values(["Date","Ticker"]).reset_index(drop=True)
    tr,va,te=frames["train"],frames["validation"],frames["test"]
    features=[]
    for c in tr.columns:
        if c in EXCLUDED: continue
        if any(t in str(c).lower() for t in LEAK):
            logging.warning("Excluding possible leakage column: %s",c); continue
        if pd.api.types.is_numeric_dtype(tr[c]): features.append(c)
    if not features: raise ValueError("No numeric features found.")
    ticker_cols=[f"Ticker_{t}" for t in sorted(set(tr.Ticker)|set(va.Ticker)|set(te.Ticker))]
    def X(d):
        a=d[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        b=pd.get_dummies(d.Ticker.astype(str),prefix="Ticker",dtype=float).reindex(columns=ticker_cols,fill_value=0)
        return pd.concat([a.reset_index(drop=True),b.reset_index(drop=True)],axis=1)
    Xtr,Xva,Xte=map(X,[tr,va,te])
    ytr=pd.to_numeric(tr[TARGET],errors="coerce").to_numpy()
    yva=pd.to_numeric(va[TARGET],errors="coerce").to_numpy()
    yte=pd.to_numeric(te[TARGET],errors="coerce").to_numpy()
    seed=int(cfg["project"]["random_seed"])
    def pipe(clf,scale=False):
        steps=[("imputer",SimpleImputer(strategy="median",add_indicator=True))]
        if scale: steps.append(("scaler",StandardScaler()))
        steps.append(("classifier",clf)); return Pipeline(steps)
    models={
      "Logistic Regression":pipe(LogisticRegression(C=.15,class_weight="balanced",max_iter=1500,random_state=seed),True),
      "Random Forest":pipe(RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=15,max_features=.7,class_weight="balanced_subsample",random_state=seed,n_jobs=-1)),
      "Extra Trees":pipe(ExtraTreesClassifier(n_estimators=300,max_depth=9,min_samples_leaf=15,max_features=.7,class_weight="balanced",random_state=seed,n_jobs=-1)),
      "Hist Gradient Boosting":pipe(HistGradientBoostingClassifier(max_iter=180,learning_rate=.04,max_leaf_nodes=12,l2_regularization=2,random_state=seed)),
}
    reports=project_path(cfg["paths"]["reports"]); models_dir=project_path(cfg["paths"]["models"])
    reports.mkdir(parents=True,exist_ok=True); models_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; candidates=[]
    for band in [0,.001,.002,.003]:
        a,b=labels(ytr,band),labels(yva,band)
        if len(set(a))<3: logging.warning("Skip band %.4f; fewer than 3 training classes",band); continue
        base=DummyClassifier(strategy="most_frequent").fit(Xtr,a)
        rows.append({"Model":"Majority Class Baseline","Neutral_Band":band,**metric(b,base.predict(Xva))})
        for name,m in models.items():
            logging.info("Training %s, neutral band %.4f",name,band)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore"); m.fit(Xtr,a)
            r={"Model":name,"Neutral_Band":band,**metric(b,m.predict(Xva))}
            rows.append(r); candidates.append((r,m,band))
    if not candidates: raise RuntimeError("No models trained; inspect target classes.")
    valdf=pd.DataFrame(rows); valdf.to_csv(reports/"direction_validation_results.csv",index=False)
    best,m,band=sorted(candidates,key=lambda q:(q[0]["Macro_F1"],q[0]["Balanced_Accuracy"]),reverse=True)[0]
    ytest=labels(yte,band); pred=m.predict(Xte)
    base=DummyClassifier(strategy="most_frequent").fit(Xtr,labels(ytr,band)); bp=base.predict(Xte)
    joblib.dump(m,models_dir/"best_direction_classifier.joblib")
    joblib.dump({"model_name":best["Model"],"target":TARGET,"labels":LABELS,"neutral_band":float(band),"features":features,"ticker_columns":ticker_cols,"selection":"validation macro-F1 then balanced accuracy"},models_dir/"direction_classifier_metadata.joblib")
    vout=va[["Date","Ticker",TARGET]].copy(); vout["Actual_Direction"]=labels(yva,band); vout["Predicted_Direction"]=m.predict(Xva); vout.to_csv(reports/"direction_validation_predictions.csv",index=False)
    tout=te[["Date","Ticker",TARGET]].copy(); tout["Actual_Direction"]=ytest; tout["Predicted_Direction"]=pred; tout.to_csv(reports/"direction_test_predictions.csv",index=False)
    cm=pd.DataFrame(confusion_matrix(ytest,pred,labels=LABELS),index=[f"Actual_{x}" for x in LABELS],columns=[f"Predicted_{x}" for x in LABELS]); cm.to_csv(reports/"direction_test_confusion_matrix.csv")
    print("\nDIRECTION VALIDATION RESULTS"); print(valdf.sort_values(["Macro_F1","Balanced_Accuracy"],ascending=False).to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print(f"\nSelected {best['Model']} | neutral band {band:.4f}")
    print("\nHELD-OUT TEST"); print(pd.DataFrame([{"Model":best["Model"],"Neutral_Band":band,**metric(ytest,pred)},{"Model":"Majority Class Baseline","Neutral_Band":band,**metric(ytest,bp)}]).to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\nCLASSIFICATION REPORT"); print(classification_report(ytest,pred,labels=LABELS,zero_division=0)); print("\nCONFUSION MATRIX"); print(cm.to_string())
if __name__=="__main__": main()
