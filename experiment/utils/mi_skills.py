import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg') # サーバー用バックグラウンド描画設定
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
import uuid
import joblib
from django.conf import settings

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)
from ..models import LotData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


pd.set_option('future.no_silent_downcasting', True)

class DatasetIntegrationSkill:
    """
    SkillNet: 複数テーマのデータを統合し、Pandas DataFrameを生成するスキル
    [Safety担保] データが0件の場合でも安全に空のDataFrameを返す
    """
    @staticmethod
    def execute(database_ids):
        # 選択されたテーマIDに含まれるすべてのLotDataを取得
        lots = LotData.objects.filter(database_id__in=database_ids)
        
        if not lots.exists():
            return pd.DataFrame(), 0
        
        data_list = []
        for lot in lots:
            # 基本情報の抽出
            row = {
                'テーマ名': lot.database.name,
                'Lot番号': lot.lot_number,
                '実験日': lot.recorded_date.strftime('%Y/%m/%d') if lot.recorded_date else None,
            }
            # JSON形式の実験データ（動的項目）を展開して結合
            if lot.experimental_data:
                row.update(lot.experimental_data)
                
            data_list.append(row)
            
        df = pd.DataFrame(data_list)
        if 'Lot番号' in df.columns:
            # groupbyしてfirst()を取ることで、複数行に散らばったデータを
            # 空欄(NaN)を無視して1行に綺麗にまとめてくれます
            df = df.groupby('Lot番号', as_index=False).first()


        return df, len(df)



class SmartImputationSkill:
    """
    SkillNet: 欠損値のスマート補完および外れ値除去を行うスキル
    """
    @staticmethod
    def execute(df, method='knn', apply_outlier_filter=False):
        if df.empty:
            return df, 0, "データが存在しません。"

        summary_messages = []

        # 1. 空文字・ハイフンなどのパージ（★修正箇所）
        df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
        df = df.replace(['None', 'NaN', 'nan', '-', 'ー', '測定不可'], np.nan).infer_objects(copy=False)

        # 2. 完全空カラムの削除
        original_col_count = len(df.columns)
        df = df.dropna(axis=1, how='all')
        dropped_cols = original_col_count - len(df.columns)
        if dropped_cols > 0:
            summary_messages.append(f"データが1件もない {dropped_cols} 個の項目を自動削除しました。")

        # 3. 保護カラムの定義と、数値型の強制適用（★修正箇所）
        protected_cols = ['テーマ名', 'Lot番号', '実験日', '担当者', '評価メモ']
        target_cols = [col for col in df.columns if col not in protected_cols]

        for col in target_cols:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # もし数値（NaN以外）が1つでも存在するなら、その列は「数値列」として上書きする
            if numeric_series.notna().sum() > 0:
                df[col] = numeric_series

        df_numeric = df[target_cols].apply(pd.to_numeric, errors='coerce')
        numeric_cols = df_numeric.dropna(axis=1, how='all').columns

        initial_count = len(df)

        # 4. 欠損値の補完ロジック (既存機能の維持)
        if len(numeric_cols) > 0:
            if method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=3)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                summary_messages.append("k-NN(最近傍探索)により欠損値をスマート補完しました。")
            elif method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                summary_messages.append("欠損値を中央値で補完しました。")
            elif method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                summary_messages.append("欠損値を平均値で補完しました。")
            elif method == 'drop':
                df = df.dropna(subset=numeric_cols)
                summary_messages.append("欠損値を含む行を削除しました。")

        # 5. 外れ値の除去 (既存機能の維持)
        if apply_outlier_filter and len(numeric_cols) > 0:
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if pd.notna(std) and std > 0:
                    lower_bound = mean - (3 * std)
                    upper_bound = mean + (3 * std)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            dropped_by_outlier = initial_count - len(df)
            if dropped_by_outlier > 0:
                summary_messages.append(f"統計的な外れ値(3σ)を {dropped_by_outlier} 件除外しました。")

        final_count = len(df)
        summary_text = " / ".join(summary_messages) if summary_messages else "前処理が完了しました。"

        return df, final_count, summary_text

class EDA_Skill:
    """
    SkillNet: 探索的データ解析(EDA)を実行するスキル
    相関行列の計算と、可視化用データの抽出を行う
    """
    @staticmethod
    # ＝＝＝ ★修正: 呼び出し元に合わせて引数 (df, target_col, feature_cols) を受け取るように変更 ＝＝＝
    def calculate_correlation(df, target_col, feature_cols):
        """指定された項目間の相関行列を計算"""
        
        target_list = [target_col] if target_col else []
        cols_to_use = target_list + feature_cols
        
        # dfに存在する項目だけにフィルタリング
        valid_cols = [col for col in cols_to_use if col in df.columns]
        
        # その中から数値データだけを抽出
        df_numeric = df[valid_cols].select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            return None
        
        # ＝＝＝ 計算不能(NaN)になったセルを 0(無相関) に置き換える ＝＝＝
        corr_matrix = df_numeric.corr().round(3).fillna(0)
        # Plotlyが扱いやすい形式（辞書リスト）に変換
        z = corr_matrix.values.tolist()
        x = corr_matrix.columns.tolist()
        y = corr_matrix.index.tolist()
        return {"z": z, "x": x, "y": y}



    @staticmethod
    def get_plot_data(df, target_col, feature_cols):
        """指定された変数間のプロットデータを抽出"""
        plots = []
        
        # カラム名が重複している場合、最初の1列だけを残す（データ側の安全装置）
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        for col in feature_cols:
            # ＝＝＝ ★修正: XとYが同じ項目だった場合は、意味がないのでスキップする ＝＝＝
            if col == target_col:
                continue
                
            if col not in df.columns or target_col not in df.columns:
                continue
            
            # 有効なデータのみ抽出
            temp_df = df[[col, target_col]].dropna()
            
            # データ型の判定
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            plots.append({
                "column_name": col,
                "x_data": temp_df[col].tolist(),
                "y_data": temp_df[target_col].tolist(),
                "type": "scatter" if is_numeric else "box"
            })
        return plots

class ModelTrainingSkill:
    """
    SkillNet: 回帰・分類を自動判定し、Plotly用のデータ抽出と
    Yellowbrickによる高度診断画像（残差プロット/混同行列）、評価コメントを生成する万能スキル
    """
    @staticmethod
    # ★修正1: 引数に hyperparams=None を追加
    def execute(df, target_col, feature_cols, algorithm='rf', label_mappings=None, hyperparams=None):
        # 1. データのクレンジング
        df = df.loc[:, ~df.columns.duplicated()].copy()
        valid_features = [col for col in feature_cols if col in df.columns and col != target_col]
        if not valid_features or target_col not in df.columns:
            return {"status": "error", "message": "指定された項目が存在しません。"}

        df_clean = df.dropna(subset=[target_col]).copy()
        if len(df_clean) < 10:
            return {"status": "error", "message": "データが少なすぎます（10件以上必要です）。"}

        X_raw = df_clean[valid_features]
        y = df_clean[target_col]

        categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns
        X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int).fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # 2. タスク（回帰か分類か）の自動判定
        is_classification = False
        if (label_mappings and target_col in label_mappings) or algorithm == 'logistic':
            is_classification = True

        # ★修正2: 送られてきたパラメータを安全に辞書として扱う
        hp = hyperparams or {}

        # 3. アルゴリズムの自動割り当て (過去のパラメータ設定と完全一致 ＋ 手動設定の反映)
        if is_classification:
            task_type = 'classification'
            if algorithm in ['rf', 'poly', 'bayesian']: 
                model = RandomForestClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)), 
                    max_depth=int(hp.get('max_depth', 10)) if hp.get('max_depth') else None,
                    random_state=42
                )
            elif algorithm == 'logistic': 
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif algorithm == 'gbr': 
                model = GradientBoostingClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 3)) if hp.get('max_depth') else 3,
                    random_state=42
                )
            else: 
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            task_type = 'regression'
            if algorithm == 'rf': 
                model = RandomForestRegressor(
                    n_estimators=int(hp.get('n_estimators', 100)), 
                    max_depth=int(hp.get('max_depth', 10)) if hp.get('max_depth') else None,
                    random_state=42
                )
            elif algorithm == 'linear': 
                model = LinearRegression()
            elif algorithm == 'ridge': 
                model = Ridge(alpha=float(hp.get('alpha', 1.0)))
            elif algorithm == 'lasso': 
                model = Lasso(alpha=float(hp.get('alpha', 0.1)))
            elif algorithm == 'gbr': 
                model = GradientBoostingRegressor(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 3)) if hp.get('max_depth') else 3,
                    random_state=42
                )
            elif algorithm == 'bayesian': 
                from sklearn.linear_model import BayesianRidge
                model = BayesianRidge()
            elif algorithm == 'poly': 
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import Pipeline
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=int(hp.get('degree', 2)))), 
                    ('linear', LinearRegression())
                ])
            else: 
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

        # 4. モデルの学習と予測（Plotlyグラフやスコア計算のため、Yellowbrickの前に実行）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # 5. 評価スコアとAI評価コメントの生成
        if is_classification:
            score1 = accuracy_score(y_test, y_pred) # 正解率
            score2 = f1_score(y_test, y_pred, average='weighted') # F1スコア
            if score1 >= 0.8: eval_msg = "素晴らしい精度です！高い確率で正しく分類できています。混同行列（下図）を見て、特定のエラーに偏りがないか確認してください。"
            elif score1 >= 0.6: eval_msg = "ある程度の分類はできていますが、誤判定が混ざっています。特徴量の見直しや、データの追加を検討してください。"
            else: eval_msg = "精度が不足しています。予測がランダムに近い状態です。結果に直結する強い特徴量（X）が足りていない可能性があります。"
        else:
            score1 = r2_score(y_test, y_pred) # R2スコア
            score2 = mean_absolute_error(y_test, y_pred) # MAE
            if score1 >= 0.8: eval_msg = "非常に高い予測精度です！残差プロット（下図）の点がゼロを中心にランダムに散らばっていれば、実運用に移行できる品質です。"
            elif score1 >= 0.5: eval_msg = "大まかな傾向は捉えていますが、予測にややブレがあります。残差プロットにU字などの偏りがある場合は、非線形アルゴリズムへの変更が有効です。"
            else: eval_msg = "予測モデルとして適合していません。現在の条件（X）だけでは結果（Y）を説明しきれないか、ノイズが多すぎる状態です。"

        # 6. Yellowbrickによる画像生成 (残差プロット or 混同行列のみ)
        img_plot1 = None
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # フォント設定 (japanize_matplotlib が入っていれば自動で日本語化されます)
            plt.rcParams['font.family'] = 'sans-serif' 

            if is_classification:
                from yellowbrick.classifier import ConfusionMatrix
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                
                # ★修正: クラスラベルを明示的に指定し、fit()を実行してYellowbrickの内部状態を同期する
                classes = list(model.classes_) if hasattr(model, 'classes_') else None
                vis1 = ConfusionMatrix(model, ax=ax1, classes=classes)
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    vis1.fit(X_train, y_train) # ★追加: これが抜けていたためエラーが発生していました
                    vis1.score(X_test, y_test)
                    
                vis1.finalize()
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
                img_plot1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
                plt.close(fig1)
            else:
                from yellowbrick.regressor import ResidualsPlot
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                vis1 = ResidualsPlot(model, ax=ax1)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    vis1.fit(X_train, y_train)
                    vis1.score(X_test, y_test)
                vis1.finalize()
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='png', bbox_inches='tight', dpi=100)
                img_plot1 = base64.b64encode(buf1.getvalue()).decode('utf-8')
                plt.close(fig1)
        except Exception as e:
            import traceback
            print("Yellowbrick Error:", traceback.format_exc())

        # 7. Plotly用の特徴量重要度の抽出 (エラー回避のため scikit-learn の機能で直接抽出)
        importances = np.zeros(len(valid_features))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 線形モデルなどの場合（分類時の多次元配列にも対応）
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                importances = coefs.mean(axis=0)[:len(valid_features)]
            else:
                importances = coefs.flatten()[:len(valid_features)]
        
        # ソートして上位15件を抽出（特徴量が3つしかない場合でもエラーになりません）
        importance_data = sorted(zip(valid_features, importances), key=lambda x: x[1], reverse=True)[:15]

        # ＝＝＝ ★ここから追加実装：モデルのPKLファイル一時保存 ＝＝＝
        save_dir = os.path.join(settings.MEDIA_ROOT, 'mi_models')
        os.makedirs(save_dir, exist_ok=True)
        temp_filename = f"model_temp_{uuid.uuid4().hex[:8]}.pkl"
        temp_file_path = os.path.join(save_dir, temp_filename)
        
        try:
            joblib.dump(model, temp_file_path)
            saved_temp_path = f"mi_models/{temp_filename}"
        except Exception as e:
            import traceback
            print("Model Save Error:", traceback.format_exc())
            saved_temp_path = None
        # ＝＝＝ ここまで追加実装 ＝＝＝

        # 8. フロントエンドへ返すデータの構築
        return {
            "status": "success",
            "task_type": task_type,
            "evaluation_message": eval_msg,
            "temp_file_path": saved_temp_path, # ★この1行を追加
            "metrics": {
                "score1": round(float(score1), 3),
                "score2": round(float(score2), 3),
                "r2": round(float(score1), 3),   
                "mae": round(float(score2), 3),
                "algorithm": algorithm,
                "hyperparams": hp,  
                "feature_importances_names": [k for k, v in importance_data],
                "feature_importances_scores": [v for k, v in importance_data]
            },
            "plots": {
                "yellowbrick_img": img_plot1
            },
            "actual_vs_predicted": {
                "actual": y_test.tolist(),
                "predicted": y_pred.tolist()
            }
        }


class InverseDesignSkill:
    """
    SkillNet: モンテカルロ法を用いた配合の逆解析（最適化）スキル
    [Safety] 学習データのMin-Max範囲内でのみ探索を行い、非現実的なレシピを防止
    """
    @staticmethod
    def run_simulation(df, target_col, feature_cols, target_value, n_trials=10000):
        # 1. モデルの再学習（Phase 3と同じロジックで瞬間的に構築）
        df_unique = df.loc[:, ~df.columns.duplicated()].copy()
        df_clean = df_unique.dropna(subset=[target_col]).copy()

        # ＝＝＝feature_cols から target_col を強制除外する安全装置 ＝＝＝
        valid_features = [col for col in feature_cols if col in df_unique.columns and col != target_col]
        
        if not valid_features:
            return {"status": "error", "message": "有効な説明変数がありません。"}
        
        X_raw = df_clean[valid_features]
        y = df_clean[target_col]
        
        categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns
        X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)
        X_encoded = X_encoded.fillna(0)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_encoded, y)

        # 2. 仮想レシピ（ランダムデータ）の生成
        # 数値項目はMin-Max、カテゴリ項目は存在する選択肢からランダムに抽出
        sim_data = []
        for _ in range(n_trials):
            row = {}
            for col in valid_features:
                if pd.api.types.is_numeric_dtype(X_raw[col]):
                    c_min, c_max = X_raw[col].min(), X_raw[col].max()
                    row[col] = np.random.uniform(c_min, c_max)
                else:
                    row[col] = np.random.choice(X_raw[col].unique())
            sim_data.append(row)
        
        df_sim = pd.DataFrame(sim_data)
        
        # 3. 仮想レシピの前処理（学習時と同じ列構造に合わせる）
        X_sim_encoded = pd.get_dummies(df_sim, columns=categorical_cols, drop_first=True, dtype=int)
        # 学習時に存在した列を確保（欠けているダミー変数は0で埋める）
        X_sim_encoded = X_sim_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # 4. 予測とスコアリング
        df_sim['predicted_y'] = model.predict(X_sim_encoded)
        # 目標値との絶対誤差を計算
        df_sim['error'] = (df_sim['predicted_y'] - target_value).abs()
        
        # 5. 誤差が小さい順にTop 5を抽出
        top_recipes = df_sim.sort_values('error').head(5)
        
        return {
            "status": "success",
            "target_value": target_value,
            "recipes": top_recipes.drop(columns=['error']).round(3).to_dict(orient='records')
        }

        # === experiment/utils/mi_skills.py の一番下に追加 ===

class DataCurationSkill:
    """
    SkillNet: 人間が定義したスキーマ（型、除外、補完ルール）に従ってデータを一括浄化するスキル
    """
    @staticmethod
    def execute(df, x_cols, y_cols, type_overrides, min_max_rules, imputation_method):
        summary_messages = []
        initial_count = len(df)

        # 1. カラムの絞り込み (除外リストのパージ)
        protected_cols = ['テーマ名', 'Lot番号', '実験日', '担当者', '評価メモ']
        target_cols = x_cols + y_cols
        keep_cols = [c for c in protected_cols if c in df.columns] + [c for c in target_cols if c in df.columns]
        keep_cols = list(dict.fromkeys(keep_cols)) # 順序維持で重複排除
        df = df[keep_cols].copy()

        # 2. 型の強制適用とハイフン等のパージ
        numeric_cols = []
        categorical_cols = []

        for col in target_cols:
            if col not in df.columns: continue
            
            # ユーザーが指定した属性を取得 (デフォルトは数値)
            col_type = type_overrides.get(col, 'numeric')

            if col_type == 'numeric':
                # 数値強制: 変換できない文字(ハイフンや空文字)は自動的にNaNになる
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            else:
                # カテゴリ強制: ハイフン等はNaNにしつつ、文字として維持する
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
                df[col] = df[col].replace(['None', 'NaN', 'nan', '-', 'ー', '測定不可'], np.nan).infer_objects(copy=False)
                categorical_cols.append(col)

        # 3. Min/Maxによる行の除外
        for rule in min_max_rules:
            col = rule.get('col')
            min_val = rule.get('min')
            max_val = rule.get('max')

            if col in numeric_cols:
                if min_val not in [None, '']: df = df[df[col] >= float(min_val)]
                if max_val not in [None, '']: df = df[df[col] <= float(max_val)]

        dropped_by_minmax = initial_count - len(df)
        if dropped_by_minmax > 0:
            summary_messages.append(f"範囲外のデータ {dropped_by_minmax} 件を除外しました。")

        # 4. 欠損値の補完 (数値)
        if len(numeric_cols) > 0:
            if imputation_method == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=3)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                summary_messages.append("k-NNにより数値の欠損を補完しました。")
            elif imputation_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                summary_messages.append("欠損値を中央値で補完しました。")
            elif imputation_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                summary_messages.append("欠損値を平均値で補完しました。")
            elif imputation_method == 'drop':
                df = df.dropna(subset=numeric_cols)
                summary_messages.append("欠損を含む行を削除しました。")
# カテゴリ変数の欠損値補完 (最頻値)
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].isnull().any():
                    # ① NaNを除外し、安全な文字列(str)に変換してからカウント
                    valid_series = df[col].dropna().astype(str)
                    
                    if len(valid_series) > 0:
                        mode_val = valid_series.value_counts().index[0]
                    else:
                        mode_val = 'Unknown'
                    
                    # ② 万が一リストや配列になっていても、強制的に最初の要素だけを抽出
                    if isinstance(mode_val, (list, tuple, set)):
                        mode_val = list(mode_val)[0]
                        
                    # ③ 確実に「単一の文字（スカラー）」に固定して穴埋めする
                    df[col] = df[col].fillna(value=str(mode_val))

        # 5. カテゴリ変数のラベル数値化 (AI学習用)
        label_mappings = {}
        if len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                # ユーザーが確認できるようにマッピング辞書を作成 (例: {"テスト": 1, "半S-400": 0})
                label_mappings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            summary_messages.append("カテゴリ項目をAI用にラベル数値化しました。")

        final_count = len(df)
        summary_text = " / ".join(summary_messages) if summary_messages else "データ浄化が完了しました。"
        
        return df, final_count, summary_text, label_mappings