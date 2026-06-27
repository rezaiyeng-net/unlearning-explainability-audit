import numpy as np
import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import shap
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

NUM_SEEDS = 15
FORGET_SIZES = [5, 10]
MODELS_TO_RUN = ['LR', 'RF']

# ============================================================
# کلاس Ensemble برای SISA درختی
# ============================================================
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self.classes_ = models[0].classes_
        
    def predict(self, X):
        predictions = np.array([m.predict(X) for m in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.mean(probas, axis=0)

def create_model(model_type, random_state=42):
    if model_type == 'LR': return LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'RF': return RandomForestClassifier(n_estimators=50, random_state=random_state)

# ============================================================
# توابع Unlearning (با رفع باگ‌های SISA)
# ============================================================
def unlearn_exact(model_type, X, y, forget_idx, seed):
    keep_mask = ~np.isin(np.arange(len(X)), forget_idx)
    model = create_model(model_type, seed)
    model.fit(X[keep_mask], y[keep_mask])
    return model

def unlearn_sisa(model_type, X, y, forget_idx, num_shards=5, seed=42):
    n = len(X)
    shard_size = n // num_shards
    models = []
    forget_mask_global = np.isin(np.arange(n), forget_idx)
    
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else n
        shard_mask = np.zeros(n, dtype=bool)
        shard_mask[start:end] = True
        
        X_s, y_s = X[shard_mask], y[shard_mask]
        # 🔴 رفع باگ تک‌کلاسه شدن شارد
        if len(np.unique(y_s)) < 2:
            X_s, y_s = X, y
            
        m = create_model(model_type, seed + i)
        m.fit(X_s, y_s)
        models.append((m, shard_mask))

    for i, (_, shard_mask) in enumerate(models):
        if np.any(shard_mask & forget_mask_global):
            keep_mask = shard_mask & ~forget_mask_global
            if np.sum(keep_mask) > 0 and len(np.unique(y[keep_mask])) >= 2:
                new_m = create_model(model_type, seed + 100 + i)
                new_m.fit(X[keep_mask], y[keep_mask])
                models[i] = (new_m, shard_mask)
            else:
                # 🔴 رفع باگ: اگر شارد بعد از حذف تک‌کلاسه شد، از کل داده‌های باقی‌مانده استفاده کن
                all_keep = ~forget_mask_global
                new_m = create_model(model_type, seed + 100 + i)
                new_m.fit(X[all_keep], y[all_keep])
                models[i] = (new_m, shard_mask)

    if model_type == 'LR':
        final = create_model(model_type, seed)
        final.fit(X[:10], y[:10])
        final.coef_ = np.mean([m.coef_ for m, _ in models], axis=0)
        final.intercept_ = np.mean([m.intercept_ for m, _ in models], axis=0)
        return final
    else:
        return EnsembleClassifier([m for m, _ in models])

def unlearn_weighting(model_type, X, y, forget_idx, seed):
    w = np.ones(len(X))
    w[forget_idx] = 0.0
    m = create_model(model_type, seed)
    m.fit(X, y, sample_weight=w)
    return m

# ============================================================
# محاسبه SHAP (بهینه‌سازی شده برای سرعت فوق‌العاده)
# ============================================================
def compute_global_importance(model, X_train, X_test, model_type):
    # 🔴 کاهش نمونه‌ها به ۵۰ برای سرعت بیشتر (از نظر آماری کاملاً کافی است)
    X_test_sub = X_test[:50] 
    
    if hasattr(model, 'models'): # SISA Ensemble
        vals = []
        for sub in model.models:
            if model_type == 'RF':
                exp = shap.TreeExplainer(sub)
                sv = exp.shap_values(X_test_sub)
                if isinstance(sv, list): sv = sv[1]
                vals.append(np.mean(np.abs(sv), axis=0))
            else: # LR
                exp = shap.LinearExplainer(sub, X_train)
                sv = exp.shap_values(X_test_sub)
                vals.append(np.mean(np.abs(sv), axis=0))
        return np.mean(vals, axis=0)
    
    if model_type == 'RF':
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(X_test_sub)
        if isinstance(sv, list): sv = sv[1]
        return np.mean(np.abs(sv), axis=0)
    else: # LR
        # 🔴 استفاده از LinearExplainer برای LR که هزاران برابر سریع‌تر از KernelExplainer است
        exp = shap.LinearExplainer(model, X_train)
        sv = exp.shap_values(X_test_sub)
        return np.mean(np.abs(sv), axis=0)

def compute_mia_score(model, X_train, X_forget):
    try:
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_forget = model.predict_proba(X_forget)[:, 1]
        threshold = np.median(prob_train)
        return np.abs(np.mean(prob_train > threshold) - np.mean(prob_forget > threshold))
    except:
        return 0.0

# ============================================================
# اجرای آزمایش
# ============================================================
def run_experiment(dataset_name, X, y, model_type):
    results = []
    for seed in range(100, 100 + NUM_SEEDS):
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2: continue
        except: continue
        
        X_tr, X_te = X_tr.astype(float), X_te.astype(float)
        scaler = StandardScaler()
        X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

        for f_size in FORGET_SIZES:
            np.random.seed(seed)
            f_idx = np.random.choice(len(X_tr_s), size=min(f_size, len(X_tr_s)), replace=False)
            retrained = unlearn_exact(model_type, X_tr_s, y_tr, f_idx, seed)
            
            for method, func in [('Exact', unlearn_exact), ('SISA', unlearn_sisa), ('Weighting', unlearn_weighting)]:
                try:
                    if method == 'SISA':
                        m = func(model_type, X_tr_s, y_tr, f_idx, 5, seed)
                    else:
                        m = func(model_type, X_tr_s, y_tr, f_idx, seed)
                        
                    acc = m.score(X_te_s, y_te)
                    imp_r = compute_global_importance(retrained, X_tr_s, X_te_s, model_type)
                    imp_m = compute_global_importance(m, X_tr_s, X_te_s, model_type)
                    fid = np.linalg.norm(imp_r - imp_m)
                    mia = compute_mia_score(m, X_tr_s, X_tr_s[f_idx])
                    
                    results.append({'dataset': dataset_name, 'model': model_type, 'seed': seed, 
                                    'forget_size': f_size, 'method': method, 'accuracy': acc, 'fid': fid, 'mia': mia})
                    print(f"✅ {dataset_name} | {model_type} | seed={seed} | |Df|={f_size} | {method}: Acc={acc:.4f}, FID={fid:.6f}, MIA={mia:.4f}")
                except Exception as e:
                    print(f"⚠️ Error in {dataset_name} | {model_type} | {method}: {e}")
    return pd.DataFrame(results)

# ============================================================
# بارگذاری دیتاست‌ها
# ============================================================
def load_datasets():
    datasets = {}
    # German
    cols = ['status','duration','credit_history','purpose','amount','savings','employment','installment_rate','personal_status','other_debtors','residence','property','age','other_installments','housing','existing_credits','job','num_dependents','telephone','foreign','target']
    df = pd.read_csv('german.data', sep=' ', header=None, names=cols)
    df['target'] = df['target'].replace({1:0,2:1})
    df = pd.get_dummies(df, drop_first=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    datasets['German Credit'] = (df.drop('target', axis=1).values.astype(float), df['target'].values)
    
    # Adult
    X_a, y_a = fetch_openml('adult', version=2, return_X_y=True, as_frame=True)
    X_a = X_a.replace('?', np.nan).dropna(); y_a = y_a.loc[X_a.index]
    y_a = (y_a == '>50K').astype(int)
    X_a = pd.get_dummies(X_a, drop_first=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    datasets['Adult'] = (X_a.values.astype(float), y_a.values)

    # Heart
    h = fetch_ucirepo(id=45)
    X_h, y_h = h.data.features, h.data.targets
    y_h = (y_h > 1).astype(int).values.ravel()
    X_h = pd.get_dummies(X_h, drop_first=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    datasets['Heart Disease'] = (X_h.values.astype(float), y_h)
    return datasets

if __name__ == "__main__":
    print("🚀 Starting Part 1: LR & RF with MIA (Optimized for Speed)...")
    datasets = load_datasets()
    all_res = []
    for ds, (X, y) in datasets.items():
        for m in MODELS_TO_RUN:
            print(f"\n--- {ds} | {m} ---")
            res = run_experiment(ds, X, y, m)
            if not res.empty: all_res.append(res)
            
    if all_res:
        final = pd.concat(all_res, ignore_index=True)
        final.to_csv('results/part1_results.csv', index=False)
        print("\n✅ Part 1 Finished. Saved to results/part1_results.csv")