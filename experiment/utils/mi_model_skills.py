import os
import uuid
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from django.conf import settings
from ..models import Database, TrainedModelMetadata
from scipy.optimize import minimize
import optuna

class MultiTargetModelTrainingSkill:
    """Phase 3 - 多彩なアルゴリズムに対応した一括学習スキル"""
    @staticmethod
    def train_and_save_models(df, database_id, target_cols, feature_cols, preprocessing_meta, models_config=None):
        # ★ SkillNet連携: 単一モデル学習エンジンを動的インポート（循環参照防止）
        from .mi_skills import ModelTrainingSkill 
        
        df = df.loc[:, ~df.columns.duplicated()].copy()
        # 1. カラム名の空白除去と、AI列の数値型変換（型保証）
        # これにより、UIから来た名前とDB上の名前の「見えないスペース」問題を解決します
        for col in feature_cols:
            if col in df.columns:
                # 数値型に強制変換（数値化できない文字列は0にする）
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 2. 列の照合
        valid_features = [col for col in feature_cols if col in df.columns]
        
        if not valid_features:
            return {"status": "error", "message": f"有効な説明変数(X)が見つかりません。対象: {feature_cols}"}
        
        try:
            database = Database.objects.get(id=database_id)
        except Database.DoesNotExist:
            return {"status": "error", "message": "対象のテーマ(データベース)が見つかりません。"}

        results = []

        # 分類タスク判定用のラベルマッピングを取得
        label_mappings = {}
        if isinstance(preprocessing_meta, dict):
            label_mappings = preprocessing_meta.get('label_mappings', {})

        for target_col in target_cols:
            if target_col not in df.columns:
                continue

            # 個別設定の取得
            current_features = valid_features
            algo_type = 'rf'
            hp = {} # ★UIからの詳細設定（ハイパーパラメータ）の受け皿
            
            if models_config and target_col in models_config:
                conf = models_config[target_col]
                if isinstance(conf, dict):
                    ui_features = conf.get('features', [])
                    if ui_features: # リストが空でなければ上書き、空なら全列(valid_features)を使う
                        current_features = [col for col in ui_features if col in df.columns]
                    algo_type = conf.get('algorithm', 'rf')
                    hp = conf.get('hyperparams', {})
                else:
                    current_features = [col for col in conf if col in df.columns]
            
            if not current_features:
                results.append({"target": target_col, "status": "error", "message": "有効な説明変数がありません。"})
                continue

            # ★ SkillNet連携: 単一モデル学習機能に処理を委譲
            # これにより、パラメータ適用、高度な前処理、Yellowbrick画像生成、評価コメントが全て自動適用されます。
            try:
                single_result = ModelTrainingSkill.execute(
                    df=df,
                    target_col=target_col,
                    feature_cols=current_features,
                    algorithm=algo_type,
                    label_mappings=label_mappings,
                    hyperparams=hp
                )
                
                if single_result.get('status') == 'success':
                    # MultiTarget固有のフォーマットに整形してフロントエンドへ返却
                    single_result['target'] = target_col
                    # フロントエンドが temp_id で正式保存処理を行うためのマッピング
                    single_result['temp_id'] = single_result.get('temp_file_path')
                    results.append(single_result)
                else:
                    results.append({
                        "target": target_col, 
                        "status": "error", 
                        "message": single_result.get('message', '学習に失敗しました。')
                    })
            except Exception as e:
                import traceback
                print(f"Multi-Target Error for {target_col}:", traceback.format_exc())
                results.append({"target": target_col, "status": "error", "message": f"エラーが発生しました: {str(e)}"})

        success_count = len([r for r in results if r['status'] == 'success'])
        return {
            "status": "success" if success_count > 0 else "error",
            "message": f"{success_count}個の目的変数のモデル学習が完了しました。",
            "details": results
        }



class MultiObjectiveInverseSkill:
    """Phase 4: Optuna（ベイズ最適化）を用いた高度な制約付き多目的最適化スキル"""
    
    @staticmethod
    def run_multi_optimization(model_ids, target_goals, df=None, fixed_features=None, mixture_settings=None):
        if fixed_features is None: fixed_features = []
        if mixture_settings is None: mixture_settings = []

        try:
            # ＝＝＝ ★修正1：[ ] などの特殊記号を消すクレンジング関数 ＝＝＝
            import re  # <--- ★この1行を追加してください！
            def clean_col_name(name):
                return re.sub(r'[\[\]<]', '', str(name))

            # 生データの列名も事前に綺麗にしておく（Min/Maxを正確に取得するため）
            if df is not None:
                df.rename(columns=lambda x: clean_col_name(x), inplace=True)

            # 画面から送られてきた制約条件の列名も綺麗にしておく
            fixed_dict = {clean_col_name(f['name']): f for f in fixed_features}
            for mix in mixture_settings:
                mix['features'] = [clean_col_name(f) for f in mix.get('features', [])]

            loaded_models = []
            all_features = set()
            
            for mid in model_ids:
                meta = TrainedModelMetadata.objects.get(id=mid)
                model = joblib.load(meta.model_file.path)

                model_class_name = model.__class__.__name__
                is_class = 'Classifier' in model_class_name or 'Logistic' in model_class_name

                # ＝＝＝ ★修正2：モデルが「実際に学習した変数」だけを内部記憶から取り出す ＝＝＝
                # （相関フィルター等で除外されたAI特徴量をここで完全に無視させます）
                if hasattr(model, 'feature_names_in_'):
                    actual_features = list(model.feature_names_in_)
                else:
                    actual_features = [clean_col_name(f) for f in meta.features_list]

                loaded_models.append({
                    'model': model,
                    'target': clean_col_name(meta.target_variable),
                    'features': actual_features, # ★11個の正しいリストがセットされる
                    'goal': float(target_goals.get(str(mid), 0)),
                    'is_classification': is_class
                })
                for f in actual_features:
                    all_features.add(f)
            
            feature_list = list(all_features)
            
            # ==========================================
            # 1. 境界条件（Bounds）の構築
            # ==========================================
            bounds_dict = {}
            
            for f in feature_list:
                nat_min, nat_max = 0.0, 100.0
                if df is not None and f in df.columns:
                    numeric_series = pd.to_numeric(df[f], errors='coerce').dropna()
                    if not numeric_series.empty:
                        nat_min, nat_max = float(numeric_series.min()), float(numeric_series.max())
                
                if f in fixed_dict:
                    rule = fixed_dict[f]
                    rtype = rule.get('type')
                    v1 = float(rule.get('val1', 0))
                    
                    if rtype == 'eq':
                        bounds_dict[f] = (v1, v1)
                    elif rtype == 'ge':
                        bounds_dict[f] = (v1, nat_max if nat_max > v1 else v1 + 100.0)
                    elif rtype == 'le':
                        bounds_dict[f] = (nat_min if nat_min < v1 else v1 - 100.0, v1)
                    elif rtype == 'range':
                        v2 = float(rule.get('val2', v1))
                        bounds_dict[f] = (v1, v2)
                else:
                    bounds_dict[f] = (nat_min, nat_max)

            # ==========================================
            # 2. Optunaの目的関数（Objective）
            # ==========================================
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                row = {}
                for f in feature_list:
                    c_min, c_max = bounds_dict[f]
                    if c_min >= c_max:
                        row[f] = c_min
                    else:
                        row[f] = trial.suggest_float(f, c_min, c_max)

                # ==========================================
                # 制約違反のペナルティ計算
                # ==========================================
                penalty = 0.0
                for mix in mixture_settings:
                    mix_feats = mix.get('features', [])
                    mix_type = mix.get('type', 'eq')
                    v1 = float(mix.get('val1', 0))
                    
                    current_sum = sum(row[f] for f in mix_feats if f in row)
                    
                    if mix_type == 'eq':
                        penalty += abs(current_sum - v1) * 1000
                    elif mix_type == 'ge' and current_sum < v1:
                        penalty += (v1 - current_sum) * 1000
                    elif mix_type == 'le' and current_sum > v1:
                        penalty += (current_sum - v1) * 1000
                    elif mix_type == 'range':
                        v2 = float(mix.get('val2', v1))
                        if current_sum < v1:
                            penalty += (v1 - current_sum) * 1000
                        elif current_sum > v2:
                            penalty += (current_sum - v2) * 1000

                # ==========================================
                # AIモデルによる予測
                # ==========================================
                df_trial = pd.DataFrame([row])
                total_error = 0.0
                
                for m in loaded_models:
                    # ＝＝＝ ★修正3：モデルが必要としている11個の特徴量だけをピンポイントで渡す ＝＝＝
                    preds = m['model'].predict(df_trial[m['features']])
                    
                    if m['is_classification']:
                        error = 0.0 if preds[0] == m['goal'] else 1.0
                    else:
                        error = abs(preds[0] - m['goal'])
                    
                    row[f"pred_{m['target']}"] = preds[0]
                    total_error += error
                
                final_score = total_error + penalty
                
                row['total_error'] = final_score
                row['penalty'] = penalty
                trial.set_user_attr("recipe", row)

                return final_score

            # ==========================================
            # 3. 探索の実行 (ベイズ最適化)
            # ==========================================
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=500)

            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            trials.sort(key=lambda t: t.value)
            
            valid_trials = [t for t in trials if t.user_attrs["recipe"].get('penalty', 0) <= 0.001]
            if not valid_trials:
                valid_trials = trials

            top_trials = valid_trials[:5]

            # ＝＝＝ NumPy型(int64等)をPython標準型に安全に変換する関数 ＝＝＝
            def to_native(val):
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, float)):
                    return round(float(val), 3)
                return val
            # ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

            sim_rows = []
            for t in top_trials:
                recipe = t.user_attrs["recipe"]
                # to_native 関数を使って全ての値を浄化する
                clean_recipe = {k: to_native(v) for k, v in recipe.items() if k != 'penalty'}
                sim_rows.append(clean_recipe)

            background_rows = []
            sample_trials = np.random.choice(trials, min(500, len(trials)), replace=False)
            for t in sample_trials:
                recipe = t.user_attrs["recipe"]
                # to_native 関数を使って全ての値を浄化する
                bg_row = {k: to_native(v) for k, v in recipe.items() if k != 'penalty'}
                background_rows.append(bg_row)

            return {
                'status': 'success',
                'results': sim_rows,
                'background_points': background_rows
            }
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}