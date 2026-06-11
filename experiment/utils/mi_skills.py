import pandas as pd
import numpy as np
import optuna
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

from ..models import LotData, Material, MaterialValue, MaterialPropertyDefinition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# ============ 1. RDKit & SMILES 変換ヘルパー ============
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')   # RDKitの大量警告ログを抑制
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

# SMILESが入っている可能性のあるプロパティキー
SMILES_KEYS = ['SMILES', 'smiles', 'Smiles', 'canonical_smiles', 'CanonicalSMILES']


def smiles_to_descriptors(smiles):
    """SMILES文字列 → 機械学習用の数値記述子dict。失敗時は空dict。"""
    if not _RDKIT_AVAILABLE:
        return {}
    if not isinstance(smiles, str) or not smiles.strip():
        return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            'SMILES_MolWt':             Descriptors.MolWt(mol),
            'SMILES_LogP':              Crippen.MolLogP(mol),
            'SMILES_TPSA':              rdMolDescriptors.CalcTPSA(mol),
            'SMILES_NumHDonors':        Lipinski.NumHDonors(mol),
            'SMILES_NumHAcceptors':     Lipinski.NumHAcceptors(mol),
            'SMILES_NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'SMILES_NumAromaticRings':  rdMolDescriptors.CalcNumAromaticRings(mol),
            'SMILES_NumRings':          rdMolDescriptors.CalcNumRings(mol),
            'SMILES_FractionCSP3':      rdMolDescriptors.CalcFractionCSP3(mol),
            'SMILES_HeavyAtomCount':    Descriptors.HeavyAtomCount(mol),
        }
    except Exception:
        return {}


# ============ 2. 独立したSMILES即時変換スキル ============
class SmilesConversionSkill:
    """
    SkillNet: 指定された列のSMILES文字列をRDKitで記述子に即座に変換し、DataFrameを拡張するスキル
    """
    @staticmethod
    def execute(df, target_col):
        if not _RDKIT_AVAILABLE:
            return df, False, "【RDKit未導入】SMILESの記述子変換には rdkit ライブラリが必要です。管理者に連絡してください。"
            
        if target_col not in df.columns:
            return df, False, f"列「{target_col}」が見つかりません。"

        # 空白や欠損値のクリーニング
        df[target_col] = df[target_col].replace(r'^\s*$', np.nan, regex=True)
        valid_smiles = df[target_col].dropna()
        
        if len(valid_smiles) == 0:
            return df, False, f"列「{target_col}」に有効なSMILESデータが含まれていません。"

        new_cols_data = {}
        for idx, val in valid_smiles.items():
            desc = smiles_to_descriptors(str(val))
            for d_key, d_val in desc.items():
                # SMILES_MolWt -> MolWt に短縮して列名を生成
                suffix = d_key.replace("SMILES_", "")
                new_col_name = f"[AI] {target_col}_{suffix}"
                
                if new_col_name not in new_cols_data:
                    new_cols_data[new_col_name] = {}
                new_cols_data[new_col_name][idx] = d_val
        
        if not new_cols_data:
             return df, False, "SMILESの変換に失敗しました（無効な形式のデータのみが含まれている可能性があります）。"

        # 抽出した記述子をDataFrameに結合
        for new_col_name, data_dict in new_cols_data.items():
            s = pd.Series(data_dict)
            df[new_col_name] = s

        # 元の文字列カラムはAIモデルでエラーになるため安全に削除
        df = df.drop(columns=[target_col])
        
        return df, True, f"「{target_col}」のSMILESから、{len(new_cols_data)}個のAI化学記述子を展開しました！"


# ==============================================================


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

        # 1. 空文字・ハイフンなどのパージ
        df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
        df = df.replace(['None', 'NaN', 'nan', '-', 'ー', '測定不可'], np.nan).infer_objects(copy=False)

        # 2. 完全空カラムの削除
        original_col_count = len(df.columns)
        df = df.dropna(axis=1, how='all')
        dropped_cols = original_col_count - len(df.columns)
        if dropped_cols > 0:
            summary_messages.append(f"データが1件もない {dropped_cols} 個の項目を自動削除しました。")

        # 3. 保護カラムの定義と、数値型の強制適用
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
    @staticmethod
    def calculate_correlation(df, target_col, feature_cols):
        """指定された項目間の相関行列を計算し、Yとの相関が強い順にソートする"""
        target_list = [target_col] if target_col else []
        cols_to_use = target_list + feature_cols
        valid_cols = [col for col in cols_to_use if col in df.columns]
        df_numeric = df[valid_cols].select_dtypes(include=[np.number])
        
        if df_numeric.empty:
            return None
        
        corr_matrix = df_numeric.corr().round(3).fillna(0)
        
        if target_col in corr_matrix.columns:
            # ターゲットとの相関の絶対値を計算し、降順でソート
            sorted_cols = corr_matrix[target_col].abs().sort_values(ascending=False).index.tolist()
            # 行と列をソートされた順番で再構築
            corr_matrix = corr_matrix.loc[sorted_cols, sorted_cols]

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
    def execute(df, target_col, feature_cols, algorithm='rf', label_mappings=None, hyperparams=None):
        # 1. データのクレンジングとバックアップ
        df = df.loc[:, ~df.columns.duplicated()].copy()

        import re
        def clean_col_name(name):
            # XGBoost等のクラッシュを防ぐため特殊記号を消去する
            return re.sub(r'[\[\]<>]', '', str(name)).strip()
        
        # 【最重要】クレンジング後の名前から「元のカッコ付きの名前」を逆引きできるマップを構築
        # これにより、SMILESや物性マスター由来の [AI] 特徴量を正確に追跡・復元します
        orig_col_map = {clean_col_name(col): col for col in df.columns}
        orig_feat_map = {clean_col_name(col): col for col in feature_cols}
        
        # データフレームの列名を安全なもの（記号なし）に一括置換
        df.rename(columns=lambda x: clean_col_name(x), inplace=True)
        
        safe_target_col = clean_col_name(target_col)
        
        # クレンジング後の世界で、実際にデータフレームに存在する特徴量を厳密に特定
        valid_safe_features = []
        for col in feature_cols:
            safe_c = clean_col_name(col)
            if safe_c in df.columns and safe_c != safe_target_col:
                valid_safe_features.append(safe_c)
                
                # 型保証: AI列または数値列として期待されるものはここで確実にfloat型へ矯正
                if '[AI]' in col or col.startswith('[AI]'):
                    df[safe_c] = pd.to_numeric(df[safe_c], errors='coerce').fillna(0)

        if not valid_safe_features or safe_target_col not in df.columns:
            return {"status": "error", "message": f"目的変数「{target_col}」または有効な説明変数（X）が見つかりません。"}

        df_clean = df.dropna(subset=[safe_target_col]).copy()
        
        if len(df_clean) < 3:
            return {
                "status": "error", 
                "message": f"有効なデータが {len(df_clean)} 件しかありません。AI学習には最低3件必要です。"
            }

        X_raw = df_clean[valid_safe_features]
        y = df_clean[safe_target_col]

        # エンコーディング（数値以外のカテゴリ変数のみOne-Hot化）
        categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
        X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int).fillna(0)
        
        # 学習用・テスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # 2. タスク（回帰か分類か）の自動判定
        is_classification = False
        if (label_mappings and target_col in label_mappings) or algorithm == 'logistic':
            is_classification = True
        
        hp = hyperparams or {}

        # 3. アルゴリズムの自動割り当て
        if is_classification:
            task_type = 'classification'
            cw = 'balanced' if hp.get('use_class_weight') == 'balanced' else None
            
            if algorithm == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 6)),
                    random_state=42, eval_metric='logloss'
                )
            elif algorithm == 'lightgbm':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 6)),
                    class_weight=cw, random_state=42, verbose=-1
                )
            elif algorithm == 'rf':
                model = RandomForestClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 10)) if hp.get('max_depth') else None,
                    class_weight=cw, random_state=42
                )
            elif algorithm == 'logistic':
                model = LogisticRegression(
                    class_weight=cw, C=float(hp.get('C', 1.0)), 
                    max_iter=int(hp.get('max_iter', 1000)), random_state=42
                )
            elif algorithm == 'gbr':
                model = GradientBoostingClassifier(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 3)), random_state=42
                )
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)

        else: # 回帰タスク
            task_type = 'regression'
            if algorithm == 'xgboost':
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 6)), random_state=42
                )
            elif algorithm == 'lightgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=int(hp.get('n_estimators', 100)),
                    max_depth=int(hp.get('max_depth', 6)), random_state=42, verbose=-1
                )
            elif algorithm == 'rf':
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
                    max_depth=int(hp.get('max_depth', 3)), random_state=42
                )
            elif algorithm == 'bayesian':
                model = BayesianRidge()
            elif algorithm == 'poly':
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=int(hp.get('degree', 2)))),
                    ('linear', LinearRegression())
                ])
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 4. モデルの学習と予測
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # 5. 評価スコアとメッセージ生成
        if is_classification:
            score1 = accuracy_score(y_test, y_pred) 
            score2 = score1  # ダミー対応
            if score1 >= 0.8: eval_msg = "素晴らしい精度です！高い確率で正しく分類できています。"
            elif score1 >= 0.6: eval_msg = "ある程度の分類はできていますが、誤判定が混ざっています。"
            else: eval_msg = "精度が不足しています。予測がランダムに近い状態です。"
        else:
            score1 = r2_score(y_test, y_pred) 
            score2 = mean_absolute_error(y_test, y_pred) 
            if score1 >= 0.8: eval_msg = "非常に高い予測精度です！実運用に移行できる品質です。"
            elif score1 >= 0.5: eval_msg = "大まかな傾向は捉えていますが、予測にややブレがあります。"
            else: eval_msg = "予測モデルとして適合していません。"

        # 6. Yellowbrickによる画像生成
        img_plot1 = None
        try:
            plt.rcParams['font.family'] = 'sans-serif' 
            if is_classification:
                from yellowbrick.classifier import ConfusionMatrix
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                classes = list(model.classes_) if hasattr(model, 'classes_') else None
                vis1 = ConfusionMatrix(model, ax=ax1, classes=classes)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    vis1.fit(X_train, y_train) 
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
            print("Yellowbrick Error:", traceback.format_exc())

        # 7. 特徴量重要度の抽出 ＆ カッコ名への完全自動復元マッピング
        raw_features_in_model = []
        importances_in_model = []
        
        if hasattr(model, 'feature_names_in_'):
            raw_features_in_model = list(model.feature_names_in_)
            if hasattr(model, 'feature_importances_'):
                importances_in_model = list(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_)
                importances_in_model = list(coefs.mean(axis=0) if coefs.ndim > 1 else coefs.flatten())
        else:
            raw_features_in_model = valid_safe_features
            importances_in_model = getattr(model, 'feature_importances_', np.zeros(len(valid_safe_features)))

        # クレンジングされた名前を、画面表示用の「カッコ付きの元の名前」に集約して復元
        restored_imp_map = {}
        for feat_name, imp_val in zip(raw_features_in_model, importances_in_model):
            # get_dummies等で増幅したダミー変数（例: AI_SMILES_LogP_low）を元のベース名に丸める
            matched_orig_name = None
            for orig_key in orig_feat_map.keys():
                if feat_name.startswith(orig_key):
                    matched_orig_name = orig_feat_map[orig_key]
                    break
            
            if not matched_orig_name:
                matched_orig_name = orig_col_map.get(feat_name, feat_name)
                
            restored_imp_map[matched_orig_name] = restored_imp_map.get(matched_orig_name, 0.0) + float(imp_val)

        importance_data = sorted(restored_imp_map.items(), key=lambda x: x[1], reverse=True)[:15]

        # モデルのPKLファイル一時保存
        save_dir = os.path.join(settings.MEDIA_ROOT, 'mi_models')
        os.makedirs(save_dir, exist_ok=True)
        temp_filename = f"model_temp_{uuid.uuid4().hex[:8]}.pkl"
        temp_file_path = os.path.join(save_dir, temp_filename)
        
        try:
            joblib.dump(model, temp_file_path)
            saved_temp_path = f"mi_models/{temp_filename}"
        except Exception as e:
            print("Model Save Error:", traceback.format_exc())
            saved_temp_path = None

        # 8. フロントエンドへ返すデータの構築
        safe_importances = [float(v) for k, v in importance_data]
        safe_importance_names = [str(k) for k, v in importance_data]
        
        if is_classification:
            safe_actual = [int(x) for x in y_test]
            safe_predicted = [int(x) for x in y_pred]
        else:
            safe_actual = [float(x) for x in y_test]
            safe_predicted = [float(x) for x in y_pred]

        return {
            "status": "success",
            "task_type": task_type,
            "evaluation_message": eval_msg,
            "temp_file_path": saved_temp_path,
            "metrics": {
                "score1": round(float(score1), 3),
                "score2": round(float(score2), 3),
                "r2": round(float(score1), 3),   
                "mae": round(float(score2), 3),
                "algorithm": algorithm,
                "hyperparams": hp,  
                "feature_importances_names": safe_importance_names,
                "feature_importances_scores": safe_importances
            },
            "plots": {
                "yellowbrick_img": img_plot1
            },
            "actual_vs_predicted": {
                "actual": safe_actual,
                "predicted": safe_predicted
            }
        }


class InverseDesignSkill:
    """
    SkillNet: Optuna（ベイズ最適化）を用いた配合の逆解析（最適化）スキル
    """
    @staticmethod
    def run_simulation(df, target_col, feature_cols, target_value, loaded_model=None, n_trials=500):
        # Optunaの探索ログを静かにする
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        df_unique = df.loc[:, ~df.columns.duplicated()].copy()
        df_clean = df_unique.dropna(subset=[target_col]).copy()

        valid_features = [col for col in feature_cols if col in df_unique.columns and col != target_col]
        if not valid_features:
            return {"status": "error", "message": "有効な説明変数がありません。"}

        X_raw = df_clean[valid_features]
        y = df_clean[target_col]

        categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns
        X_encoded_base = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int).fillna(0)
        X_encoded_columns = X_encoded_base.columns

        if loaded_model is not None:
            model = loaded_model
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_encoded_base, y)

        def objective(trial):
            row = {}
            for col in valid_features:
                if pd.api.types.is_numeric_dtype(X_raw[col]):
                    c_min, c_max = float(X_raw[col].min()), float(X_raw[col].max())
                    if c_min == c_max:
                        row[col] = c_min 
                    else:
                        row[col] = trial.suggest_float(col, c_min, c_max)
                else:
                    choices = X_raw[col].dropna().unique().tolist()
                    row[col] = trial.suggest_categorical(col, choices)

            df_trial = pd.DataFrame([row])
            X_trial_encoded = pd.get_dummies(df_trial, columns=categorical_cols, drop_first=True, dtype=int)
            X_trial_encoded = X_trial_encoded.reindex(columns=X_encoded_columns, fill_value=0)

            pred = model.predict(X_trial_encoded)[0]
            error = abs(pred - target_value)

            row['predicted_y'] = pred
            row['error'] = error
            trial.set_user_attr("recipe", row)

            return error

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trials.sort(key=lambda t: t.value)
        top_trials = trials[:5]

        recipes = []
        for t in top_trials:
            recipe = t.user_attrs["recipe"]
            clean_recipe = {k: round(v, 3) if isinstance(v, float) else v for k, v in recipe.items() if k != 'error'}
            recipes.append(clean_recipe)

        return {
            "status": "success",
            "target_value": target_value,
            "recipes": recipes
        }


class DataCurationSkill:
    """
    SkillNet: 人間が定義したスキーマ（詳細設定パネルの条件）に従ってデータを一括浄化するスキル
    """
    @staticmethod
    def execute(df, x_cols, y_cols, advanced_settings, imputation_method):
        summary_messages = []
        initial_count = len(df)

        protected_cols = ['テーマ名', 'Lot番号', '実験日', '担当者', '評価メモ']
        target_cols = x_cols + y_cols
        keep_cols = [c for c in protected_cols if c in df.columns] + [c for c in target_cols if c in df.columns]
        keep_cols = list(dict.fromkeys(keep_cols))
        df = df[keep_cols].copy()

        numeric_cols = []
        categorical_cols = []

        for col in target_cols:
            if col not in df.columns: continue
            
            settings_data = advanced_settings.get(col, {'type': 'numeric'})
            col_type = settings_data.get('type', 'numeric')
            
            # ★ Y変数はSMILESや文字化けを防ぐため、カテゴリ以外なら強制的に数値に戻す安全装置
            if col in y_cols and col_type != 'categorical':
                col_type = 'numeric'

            if col_type == 'numeric':
                # --- 数値データの場合 ---
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
                
                min_val = settings_data.get('min')
                max_val = settings_data.get('max')
                if min_val not in [None, '']: df = df[df[col] >= float(min_val)]
                if max_val not in [None, '']: df = df[df[col] <= float(max_val)]
                
            else:
                # --- カテゴリデータの場合 ---
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
                df[col] = df[col].replace(['None', 'NaN', 'nan', '-', 'ー', '測定不可'], np.nan).infer_objects(copy=False)
                categorical_cols.append(col)
                
                allowed_categories = settings_data.get('categories', [])
                if allowed_categories and len(allowed_categories) > 0:
                    df = df[df[col].astype(str).isin(allowed_categories) | df[col].isna()]

        dropped_by_filter = initial_count - len(df)
        if dropped_by_filter > 0:
            summary_messages.append(f"設定された除外条件・抽出条件により {dropped_by_filter} 件を除外しました。")

        # 3. 欠損値の補完 (数値)
        if len(numeric_cols) > 0:
            if imputation_method == 'knn':
                from sklearn.impute import KNNImputer
                empty_cols = [col for col in numeric_cols if df[col].notna().sum() == 0]
                if empty_cols:
                    return df, 0, f"error:【補完エラー】項目「{', '.join(empty_cols)}」には有効な数値がありません。この項目を除外するか、補完方法を『中央値』に変更してください。", {}
                
                if len(df) < 3:
                    return df, 0, f"error:抽出後のデータが {len(df)} 件しかありません。k-NN法での補完には少なすぎるため、補完方法を『中央値』に変更してください。", {}

                try:
                    imputer = KNNImputer(n_neighbors=min(3, len(df)))
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    summary_messages.append("k-NNにより数値の欠損を補完しました。")
                except Exception as e:
                    return df, 0, f"error:k-NN補完中にエラーが発生しました（{str(e)}）。", {}

            elif imputation_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                summary_messages.append("欠損値を中央値で補完しました。")
                
        # 4. カテゴリ変数の欠損値補完 (最頻値)
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].isnull().any():
                    valid_series = df[col].dropna().astype(str)
                    mode_val = valid_series.value_counts().index[0] if len(valid_series) > 0 else 'Unknown'
                    if isinstance(mode_val, (list, tuple, set)): mode_val = list(mode_val)[0]
                    df[col] = df[col].fillna(value=str(mode_val))

        # 5. カテゴリ変数のラベル数値化 (AI学習用)
        label_mappings = {}
        if len(categorical_cols) > 0:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                label_mappings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}
            summary_messages.append("カテゴリ項目をAI用にラベル数値化しました。")

        final_count = len(df)
        summary_text = " / ".join(summary_messages) if summary_messages else "データ浄化が完了しました。"
        
        return df, final_count, summary_text, label_mappings


class FeatureEngineeringSkill:
    @staticmethod
    # ★修正1: 引数に workspace_id=None を追加
    def execute(df_experiment, y_cols=None, workspace_id=None):
        if df_experiment.empty:
            return df_experiment, []

        new_features = []
        properties_dict = {}
        exclude_keys = ['CID', 'id', 'ID', 'cas_number', 'CAS', 'cas-number', 'CasNo']

        # 1. マスターデータのロード（ハイブリッド・フォールバック対応）
        try:
            # ヘルパー関数: 取得したマスタ群からDataFrameを生成し、SMILES解析も行う
            def build_master_df(queryset):
                data = {}
                for mat in queryset:
                    mat_name = None
                    numeric_props = {}
                    for val_obj in mat.values.all():
                        prop_key = val_obj.definition.property_key
                        # 物質名の特定
                        if prop_key in ['JapaneseName', 'EnglishName', '名前', '材料名', '物質名', '品名', '名称', 'name', 'MaterialName', 'Name']:
                            mat_name = val_obj.value
                        # SMILESの解析
                        elif prop_key in SMILES_KEYS:
                            desc = smiles_to_descriptors(val_obj.value)
                            for d_key, d_val in desc.items():
                                if np.isfinite(d_val): numeric_props[d_key] = d_val
                        # 数値データの取得（ハイフンなどはここでValueErrorになり弾かれる＝空欄扱いになる）
                        elif val_obj.definition.data_type == 'number' and prop_key not in exclude_keys:
                            try:
                                f_val = float(val_obj.value)
                                if np.isfinite(f_val): numeric_props[prop_key] = f_val
                            except (ValueError, TypeError): pass
                    
                    if mat_name and numeric_props:
                        data[mat_name] = numeric_props
                
                if not data:
                    return pd.DataFrame()
                # 物質名(mat_name)をインデックス(行名)としたDataFrameを作成
                return pd.DataFrame.from_dict(data, orient='index')

            # A. 全社共有マスタの取得
            global_materials = Material.objects.filter(workspace__isnull=True).prefetch_related('values__definition')
            df_global = build_master_df(global_materials)

            # B. ワークスペース固有(ローカル)マスタの取得
            df_local = pd.DataFrame()
            if workspace_id:
                local_materials = Material.objects.filter(workspace_id=workspace_id).prefetch_related('values__definition')
                df_local = build_master_df(local_materials)

            # C. 究極のハイブリッド・フォールバック結合
            if df_local.empty:
                df_master = df_global.copy()
            elif df_global.empty:
                df_master = df_local.copy()
            else:
                # ローカルをベースに、空欄(NaN)をグローバルで埋める神機能
                df_master = df_local.combine_first(df_global)

            # D. 既存の処理に繋ぐため、マージ完了後のDataFrameを辞書(properties_dict)に戻す
            if not df_master.empty:
                for mat_name, row in df_master.iterrows():
                    # NaN(空欄)を除外してクリーンな辞書を作成
                    clean_row = {k: v for k, v in row.items() if pd.notna(v)}
                    if clean_row:
                        properties_dict[mat_name] = clean_row

        except Exception as e:
            import traceback
            print("Master Data Error:", traceback.format_exc())
            return df_experiment, []

        # 2. 特徴量生成（ハイブリッド方式）
        original_cols = df_experiment.columns.tolist()
        
        # --- パターンA: 列名に直接材料名が入っている場合 ---
        for col in original_cols:
            for mat_name, props in properties_dict.items():
                if mat_name in col:
                    weight_series = pd.to_numeric(df_experiment[col], errors='coerce').fillna(0)
                    for prop_name, prop_val in props.items():
                        new_col_name = f"[AI] {mat_name}_{prop_name}寄与度"
                        if new_col_name not in df_experiment.columns:
                            df_experiment[new_col_name] = 0.0
                        df_experiment[new_col_name] += weight_series * prop_val
                        if new_col_name not in new_features: new_features.append(new_col_name)

       # --- パターンB: 材料名と配合量が別々の列になっている場合 ---
        import re
        import unicodedata

        number_groups = {}
        for col in df_experiment.columns:
            norm_col = unicodedata.normalize('NFKC', col)
            num_match = re.search(r'\d+(?![%％度])', norm_col)
            
            if num_match:
                num = num_match.group()
                if num not in number_groups:
                    number_groups[num] = []
                number_groups[num].append(col)

        for num, cols in number_groups.items():
            if len(cols) < 2:
                continue 
                
            mat_col = None
            amt_col = None
            
            for c in cols:
                if any(keyword in c for keyword in ['量', 'g', '%', 'ml', '部', '比', '添加']):
                    amt_col = c
                else:
                    mat_col = c
            
            if mat_col and amt_col:
                for idx, row in df_experiment.iterrows():
                    mat_name = str(row[mat_col]).strip() if pd.notna(row[mat_col]) else ""
                    amount = pd.to_numeric(row[amt_col], errors='coerce')
                    
                    if mat_name in properties_dict and pd.notna(amount):
                        props = properties_dict[mat_name]
                        for prop_name, prop_val in props.items():
                            new_col_name = f"[AI] {prop_name}寄与度"
                            
                            if new_col_name not in df_experiment.columns:
                                df_experiment[new_col_name] = 0.0
                                if new_col_name not in new_features: new_features.append(new_col_name)
                            
                            df_experiment.loc[idx, new_col_name] += (amount * prop_val)

       # 3. 相関による自動フィルタリング
        if new_features and y_cols:
            df_temp = df_experiment.copy()
            df_y = df_temp[y_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            df_ai = df_temp[new_features].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            corrs_dict = {}
            for y_col in y_cols:
                corrs_dict[y_col] = df_ai.corrwith(df_y[y_col], method='pearson').abs()
            
            corrs_df = pd.DataFrame(corrs_dict)
            max_corrs = corrs_df.max(axis=1)

            threshold = 0.05
            
            # ==========================================
            # 相関が低すぎる、または変動がない変数を削除
            # ==========================================
            keep_features = []
            
            # RuntimeWarning (0での割り算など) を非表示にする
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                
                # ★修正箇所：target_col ではなく y_cols を使って全ての目的変数との相関を確認する
                numeric_df = df_experiment.select_dtypes(include=[np.number])
                max_corrs = pd.Series(0.0, index=numeric_df.columns)
                
                if y_cols:
                    # y_cols の中から数値列として存在するものを抽出
                    valid_y_cols = [y for y in y_cols if y in numeric_df.columns]
                    for y_c in valid_y_cols:
                        corrs = numeric_df.corrwith(numeric_df[y_c]).abs()
                        # 複数のYがある場合、最も高い相関スコアを採用してAI列を生き残らせる
                        max_corrs = np.maximum(max_corrs, corrs.fillna(0))

                for feat in new_features:
                    score = max_corrs.get(feat, 0)
                    if pd.isna(score): 
                        score = 0  
                    
                    if score > threshold:
                        keep_features.append(feat)
                
                drop_features = [f for f in new_features if f not in keep_features]
                
                if drop_features:
                    df_experiment.drop(columns=drop_features, inplace=True)
                    new_features = keep_features

                # ==========================================
                # 多重共線性（似たような変数）の排除
                # ==========================================
                if len(new_features) > 1:
                    # ここでも警告が出ないように計算
                    feature_corr = df_experiment[new_features].corr().abs()
                    upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
                    
                    to_drop_multi = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.95)]
                    
                    if to_drop_multi:
                        df_experiment.drop(columns=to_drop_multi, inplace=True)
                        new_features = [f for f in new_features if f not in to_drop_multi]
        # 4. 標準化
        if new_features:
            scaler = StandardScaler()
            df_experiment[new_features] = scaler.fit_transform(df_experiment[new_features])

        return df_experiment, new_features