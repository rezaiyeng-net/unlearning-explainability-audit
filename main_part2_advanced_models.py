import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import shap
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)

# ====================== Helper functions and data loading ======================
def load_datasets():
    datasets = {}
    
    # 1. German Credit
    cols = ['status','duration','credit_history','purpose','amount','savings',
            'employment','installment_rate','personal_status','other_debtors',
            'residence','property','age','other_installments','housing',
            'existing_credits','job','num_dependents','telephone','foreign','target']
    df = pd.read_csv('german.data', sep=' ', header=None, names=cols)
    df['target'] = df['target'].replace({1: 0, 2: 1})
    df = pd.get_dummies(df, drop_first=True)
    
    X_g = df.drop('target', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    y_g = df['target'].values
    datasets['german'] = (X_g, y_g)

    # 2. Adult
    try:
        X_a, y_a = fetch_openml('adult', version=2, return_X_y=True, as_frame=True)
        X_a = X_a.replace('?', np.nan).dropna()
        y_a = y_a.loc[X_a.index]
        y_a = (y_a == '>50K').astype(int)
        X_a = pd.get_dummies(X_a, drop_first=True)
        X_a = X_a.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        datasets['adult'] = (X_a, y_a.values)
    except Exception as e:
        print(f"⚠️ Adult error: {e}")

    # 3. Heart Disease
    try:
        h = fetch_ucirepo(id=45)
        X_h, y_h = h.data.features, h.data.targets
        y_h = (y_h > 1).astype(int).values.ravel()
        X_h = pd.get_dummies(X_h, drop_first=True)
        X_h = X_h.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        datasets['heart'] = (X_h, y_h)
    except Exception as e:
        print(f"⚠️ Heart error: {e}")

    return datasets

# ====================== Main class and functions ======================
class EnsembleClassifier:
    def __init__(self, models):
        self.models = models
        self.classes_ = np.array([0, 1])
        
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.round(preds.mean(axis=0)).astype(int)

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.models])
        return probs.mean(axis=0)

def create_model(model_type):
    if model_type == 'GB':
        return GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=3)
    elif model_type == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=False)
    else:
        raise ValueError("Unsupported model type")

def unlearn_sisa(model_type, X, y, forget_idx, num_shards=5, seed=42):
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    shard_size = n // num_shards
    models = []
    
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else n
        shard_idx = indices[start:end]
        
        if any(idx in forget_idx for idx in shard_idx):
            train_idx = np.setdiff1d(shard_idx, forget_idx)
            if len(train_idx) == 0: continue
            if len(np.unique(y[train_idx])) < 2: train_idx = np.setdiff1d(indices, forget_idx)
            model = create_model(model_type)
            model.fit(X[train_idx], y[train_idx])
            models.append(model)
        else:
            if len(np.unique(y[shard_idx])) >= 2:
                model = create_model(model_type)
                model.fit(X[shard_idx], y[shard_idx])
                models.append(model)
            else:
                model = create_model(model_type)
                model.fit(X, y)
                models.append(model)

    if len(models) == 0: return create_model(model_type)
    return EnsembleClassifier(models)

def compute_global_importance(model, X_train, X_test, model_type):
    # Convert to numpy to avoid dimensional inconsistency
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)[:200] # Limit for speed
    
    def extract_importance(shap_vals):
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            sv = np.array(shap_vals)
            if sv.ndim == 3: sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        return np.abs(sv).mean(axis=0)

    if hasattr(model, 'models'):  # SISA Ensemble
        vals = []
        for m in model.models:
            if model_type == 'GB':
                explainer = shap.TreeExplainer(m)
                shap_vals = explainer.shap_values(X_test_np)
            else: # MLP
                background = shap.kmeans(X_train_np, 10)
                explainer = shap.KernelExplainer(m.predict_proba, background)
                shap_vals = explainer.shap_values(X_test_np)
            vals.append(extract_importance(shap_vals))
        return np.mean(vals, axis=0) if vals else np.zeros(X_test_np.shape[1])
    else:
        if model_type == 'GB':
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_np)
        else: # MLP
            background = shap.kmeans(X_train_np, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_test_np)
        return extract_importance(shap_vals)

def compute_fid(unlearned_model, retrained_model, X_train, X_test, model_type):
    imp_un = compute_global_importance(unlearned_model, X_train, X_test, model_type)
    imp_re = compute_global_importance(retrained_model, X_train, X_test, model_type)
    return np.linalg.norm(imp_un - imp_re)

def compute_mia_score(model, X_train, X_forget):
    try:
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_forget = model.predict_proba(X_forget)[:, 1]
        threshold = np.median(prob_train)
        return np.abs(np.mean(prob_train > threshold) - np.mean(prob_forget > threshold))
    except:
        return 0.0

# ====================== Running the experiment ======================
NUM_SEEDS = 15
FORGET_SIZES = [5, 10]

def run_experiment(dataset_name, X, y, model_type):
    results = []
    for seed in range(100, 100 + NUM_SEEDS):
        for forget_size in FORGET_SIZES:
            np.random.seed(seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
            
            forget_idx = np.random.choice(len(X_train), size=forget_size, replace=False)
            
            # Exact
            exact_model = create_model(model_type)
            keep_mask = ~np.isin(np.arange(len(X_train)), forget_idx)
            exact_model.fit(X_train[keep_mask], y_train[keep_mask])
            
            # SISA
            sisa_model = unlearn_sisa(model_type, X_train, y_train, forget_idx, seed=seed)
            
            # Weighting
            weighting_model = create_model(model_type)
            sample_weights = np.ones(len(y_train))
            sample_weights[forget_idx] = 0.0
            weighting_model.fit(X_train, y_train, sample_weight=sample_weights)
            
            for name, model in [('Exact', exact_model), ('SISA', sisa_model), ('Weighting', weighting_model)]:
                try:
                    acc = accuracy_score(y_test, model.predict(X_test))
                    fid = compute_fid(model, exact_model, X_train, X_test, model_type)
                    mia = compute_mia_score(model, X_train, X_train[forget_idx])
                    
                    results.append({
                        'dataset': dataset_name, 'model': model_type, 'seed': seed,
                        'forget_size': forget_size, 'method': name,
                        'accuracy': round(acc, 4), 'fid': round(fid, 6), 'mia': round(mia, 4)
                    })
                    print(f"✅ {dataset_name} | {model_type} | seed={seed} | |Df|={forget_size} | {name}: Acc={acc:.4f}, FID={fid:.6f}, MIA={mia:.4f}")
                except Exception as e:
                    print(f"⚠️ Error in {name}: {e}")
    return results

if __name__ == "__main__":
    print("🚀 Starting Part 2: Gradient Boosting (GB) & MLP with MIA evaluation...")
    datasets = load_datasets()
    all_results = []
    
    for ds_name, (X, y) in datasets.items():
        for model_type in ['GB', 'MLP']:
            print(f"\n{'='*60}\nRunning {ds_name} - {model_type} ...\n{'='*60}")
            res = run_experiment(ds_name, X, y, model_type)
            all_results.extend(res)

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv('results/part2_results.csv', index=False)
        print("\n✅ Part 2 Results saved to results/part2_results.csv")
