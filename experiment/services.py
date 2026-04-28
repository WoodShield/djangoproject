import pandas as pd
import numpy as np

class ExperimentAIAnalyzer:
    """
    3シグマ（標準偏差）に基づき、単純な入力ミスや異常値を検知するエンジン
    """

    @classmethod
    def check_anomaly(cls, database, experimental_data, current_lot_id=None):
        past_lots = database.lots.all()
        if current_lot_id:
            past_lots = past_lots.exclude(id=current_lot_id)
            
        past_data = [lot.experimental_data for lot in past_lots if lot.experimental_data]
        
        # 統計的に意味のある数（最低5件程度）が集まるまではチェックしない
        if len(past_data) < 5:
            return None

        df_past = pd.DataFrame(past_data)
        df_past = df_past.apply(pd.to_numeric, errors='coerce')
        
        # 異常検知の対象として設定されている項目のみを抽出
        target_cols = [item.item_name for item in database.items.filter(use_for_anomaly=True)]
        
        # 今回の入力データを数値化
        current_numeric_data = {}
        for k, v in experimental_data.items():
            if k in target_cols:
                try:
                    current_numeric_data[k] = float(v)
                except (ValueError, TypeError):
                    pass

        # ＝＝＝ 3シグマ（単独項目チェック）のみを実行 ＝＝＝
        for col, val in current_numeric_data.items():
            if col in df_past.columns:
                mean = df_past[col].mean()
                std = df_past[col].std()
                
                if std > 0:
                    z_score = abs((val - mean) / std)
                    # 3シグマを超える（99.7%の範囲から外れる）場合に警告
                    if z_score > 3:
                        return f"【入力ミス確認】「{col}」の数値が過去の平均から大きく外れています（桁間違いの可能性）。"

        return None

    @classmethod
    def find_similar_lots(cls, database, experimental_data, current_lot_id=None):
        """
        類似検索機能は廃止（常にNoneを返す）
        """
        return None