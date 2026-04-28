import os
import uuid
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from django.conf import settings
from ..models import Database, TrainedModelMetadata

class MultiTargetModelTrainingSkill:
    """Phase 3 - 多彩なアルゴリズムに対応した一括学習スキル"""
    @staticmethod
    def train_and_save_models(df, database_id, target_cols, feature_cols, preprocessing_meta, models_config=None):
        df = df.loc[:, ~df.columns.duplicated()].copy()
        valid_features = [col for col in feature_cols if col in df.columns]
        
        if not valid_features:
            return {"status": "error", "message": "有効な説明変数(X)が存在しません。"}
        
        try:
            database = Database.objects.get(id=database_id)
        except Database.DoesNotExist:
            return {"status": "error", "message": "対象のテーマ(データベース)が見つかりません。"}

        save_dir = os.path.join(settings.MEDIA_ROOT, 'mi_models')
        os.makedirs(save_dir, exist_ok=True)

        results = []

        for target_col in target_cols:
            if target_col not in df.columns:
                continue

            # 個別設定の取得
            current_features = valid_features
            algo_type = 'rf'
            if models_config and target_col in models_config:
                conf = models_config[target_col]
                if isinstance(conf, dict):
                    current_features = [col for col in conf.get('features', []) if col in df.columns]
                    algo_type = conf.get('algorithm', 'rf')
                else:
                    current_features = [col for col in conf if col in df.columns]
            
            if not current_features:
                results.append({"target": target_col, "status": "error", "message": "有効な説明変数がありません。"})
                continue

            df_clean = df.dropna(subset=[target_col]).copy()
            
            if len(df_clean) < 10:
                results.append({"target": target_col, "status": "error", "message": f"有効な正解データが少なすぎます（{len(df_clean)}件）。"})
                continue

            X = df_clean[current_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            y = pd.to_numeric(df_clean[target_col], errors='coerce').fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # ★アルゴリズム分岐
            if algo_type == 'linear':
                model = LinearRegression()
            elif algo_type == 'poly':
                model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
            elif algo_type == 'bayesian':
                model = BayesianRidge()
            elif algo_type == 'logistic':
                # ロジスティック回帰の場合は、yを0か1に丸める（合否判定などの分類用）
                y_train_bin = (y_train > y_train.median()).astype(int)
                y_test_bin = (y_test > y_train.median()).astype(int)
                model = LogisticRegression(random_state=42)
                model.fit(X_train, y_train_bin)
            elif algo_type == 'gbr':
                model = GradientBoostingRegressor(random_state=42)
            elif algo_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif algo_type == 'lasso':
                model = Lasso(alpha=0.1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

            if algo_type != 'logistic':
                model.fit(X_train, y_train)

            # 予測と評価
            if algo_type == 'logistic':
                y_pred = model.predict(X_test)
                r2 = accuracy_score((y_test > y_train.median()).astype(int), y_pred) # 分類なのでAccuracyをR2の箱に入れる
                mae = 0 # 分類なのでMAEは0
            else:
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

            # 特徴量重要度の抽出
            importances = np.zeros(len(current_features))
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_.flatten()[:len(current_features)])
            elif algo_type == 'poly':
                importances = np.abs(model.named_steps['linear'].coef_.flatten()[:len(current_features)])

            importance_data = sorted(zip(current_features, importances), key=lambda x: x[1], reverse=True)[:15]

            # --- 修正箇所 (135行目付近) ---
            metrics = {
                "algorithm": algo_type, # ★この1行を追加
                "r2": round(float(r2), 3),
                "mae": round(float(mae), 3),
                "rmse": 0, # 計算を簡略化
                "train_size": len(X_train),
                "test_size": len(X_test),
                "feature_importances_names": [k for k, v in importance_data],
                "feature_importances_scores": [v for k, v in importance_data]
            }

            filename = f"model_database{database_id}_{uuid.uuid4().hex[:8]}.pkl"
            file_path = os.path.join(save_dir, filename)
            
            try:
                joblib.dump(model, file_path)
            except Exception as e:
                results.append({"target": target_col, "status": "error", "message": f"保存失敗: {str(e)}"})
                continue

            results.append({
                "target": target_col,
                "status": "success",
                "metrics": metrics,
                "features": current_features, 
                "filename": f"mi_models/{filename}", 
                "actual_vs_predicted": {
                    "actual": y_test.tolist(),
                    "predicted": y_pred.tolist()
                }
            })

        success_count = len([r for r in results if r['status'] == 'success'])
        return {
            "status": "success" if success_count > 0 else "error",
            "message": f"{success_count}個の目的変数のモデル学習が完了しました。",
            "details": results
        }


class MultiObjectiveInverseSkill:
    """Phase 4: 保存済みモデルをロードし、多目的最適化（逆解析）を行うスキル"""
    @staticmethod
    def run_multi_optimization(model_ids, target_goals, df=None, n_trials=20000):
        try:
            loaded_models = []
            all_features = set()
            
            for mid in model_ids:
                meta = TrainedModelMetadata.objects.get(id=mid)
                model = joblib.load(meta.model_file.path)
                loaded_models.append({
                    'model': model,
                    'target': meta.target_variable,
                    'features': meta.features_list,
                    'goal': float(target_goals.get(str(mid), 0))
                })
                for f in meta.features_list:
                    all_features.add(f)
            
            feature_list = list(all_features)
            
            feature_bounds = {}
            for f in feature_list:
                if df is not None and f in df.columns:
                    numeric_series = pd.to_numeric(df[f], errors='coerce').dropna()
                    if not numeric_series.empty:
                        feature_bounds[f] = (numeric_series.min(), numeric_series.max())
                    else:
                        feature_bounds[f] = (0, 1)
                else:
                    feature_bounds[f] = (0, 100)
            
            sim_rows = []
            for _ in range(n_trials):
                row = {}
                for f in feature_list:
                    vmin, vmax = feature_bounds[f]
                    if vmin == vmax:
                        row[f] = vmin
                    else:
                        row[f] = np.random.uniform(vmin, vmax)
                sim_rows.append(row)
            
            df_sim = pd.DataFrame(sim_rows)
            df_sim['total_error'] = 0
            
            for m_info in loaded_models:
                X_input = df_sim[m_info['features']]
                preds = m_info['model'].predict(X_input)
                df_sim[f"pred_{m_info['target']}"] = preds
                
                error = np.abs(preds - m_info['goal'])
                df_sim['total_error'] += error

            # 4. 誤差が小さい順に上位5件を抽出
            top_results = df_sim.sort_values('total_error').head(5)
            
            # ★グラフ描画用に、探索した仮想レシピの中からランダムに1000件を抽出
            background_sample = df_sim.sample(n=min(1000, len(df_sim)))
            
            return {
                'status': 'success',
                'results': top_results.to_dict(orient='records'),
                'background_points': background_sample.to_dict(orient='records')
            }
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}