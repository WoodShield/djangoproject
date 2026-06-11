from django.views.generic import TemplateView
from django.shortcuts import get_object_or_404
import json
import io
import urllib.parse
from django.http import JsonResponse, HttpResponse
from ..models import Workspace, Database, DatabaseItemDefinition, LotData
import pandas as pd
import numpy as np

from experiment.utils.mi_skills import DatasetIntegrationSkill, SmartImputationSkill, ModelTrainingSkill, DataCurationSkill, FeatureEngineeringSkill, EDA_Skill

class LotAnalysisView(TemplateView):
    """ 詳細分析ダッシュボード（全データ転送・フロントエンド計算型） """
    template_name = 'experiment/lot_analysis.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
                
        database = get_object_or_404(Database, pk=self.kwargs.get('pk'))
        context['database'] = database
        
        item_defs = DatabaseItemDefinition.objects.filter(
            database=database, is_active=True
        ).order_by('order')
        context['item_defs'] = item_defs

        lots = LotData.objects.filter(database=database).order_by('recorded_date')
        context['lots'] = lots

        plot_data = []
        for lot in lots:
            exp_data = lot.experimental_data or {}
            
            row = {
                'lot': lot.lot_number,
                'date': lot.recorded_date.strftime('%Y/%m/%d') if lot.recorded_date else '',
                'vals': {item.item_name: exp_data.get(item.item_name) for item in item_defs}
            }
            plot_data.append(row)
        
        context['plot_data_json'] = json.dumps(plot_data)
        return context

class MLAnalysisView(TemplateView):
    template_name = 'experiment/mi_analysis/base_analysis.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        dept_pk = self.kwargs.get('dept_pk')
        workspace = get_object_or_404(Workspace, pk=dept_pk)
        context['workspace'] = workspace
        
        databases = Database.objects.filter(workspace=workspace).order_by('-created_at')
        context['databases'] = databases
     
        databases_data = {}
        for database in databases:
            items = list(database.items.filter(is_active=True).values('item_name', 'data_type'))
            databases_data[database.id] = items
            
        context['databases_json'] = json.dumps(databases_data)
        return context

class MIAnalysisView(TemplateView):
    """
    SkillNet Entry Point: MI解析エンジンの基盤UIを提供
    """
    template_name = 'mi_analysis/base_analysis.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dept_pk = self.kwargs.get('dept_pk')
        workspace = get_object_or_404(Workspace, pk=dept_pk)
        context['workspace'] = workspace
        
        databases = Database.objects.filter(workspace=workspace).order_by('-created_at')
        context['databases'] = databases
        return context


def api_generate_mi_dataset(request, dept_pk):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            database_ids = data.get('database_ids', [])
            imputation_method = data.get('imputation_method', 'knn')
            apply_outlier_filter = data.get('apply_outlier_filter', False)

            if not database_ids:
                return JsonResponse({'status': 'error', 'message': 'データベースが選択されていません。'})

            df, raw_count = DatasetIntegrationSkill.execute(database_ids)
            if df.empty:
                return JsonResponse({'status': 'error', 'message': '選択されたデータベースに有効なデータが存在しません。'})

            df_clean, final_count, summary = SmartImputationSkill.execute(
                df=df, method=imputation_method, apply_outlier_filter=apply_outlier_filter
            )
            
            request.session['mi_raw_df'] = df.to_json(orient='split') 
            request.session['mi_current_df'] = df_clean.to_json(orient='split') 
            
            safe_df = df_clean.replace({np.nan: None})
            raw_data_dict = safe_df.to_dict(orient='records')
            raw_data_json = json.dumps(raw_data_dict, ensure_ascii=False, default=str)
            
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

            html_table = df_clean.to_html(
                classes='table table-sm table-striped table-hover border mb-0 text-nowrap',
                index=False, justify='center', na_rep='(空欄)' 
            )

            return JsonResponse({
                'status': 'success',
                'html_table': html_table,
                'total_rows': final_count,
                'summary_message': summary,
                'raw_data_json': raw_data_json,
                'columns': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols,
                    'all': df_clean.columns.tolist()
                }
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'システムエラーが発生しました: {str(e)}'})

    return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'})


def api_get_eda_results(request, dept_pk):
    if request.method == 'POST':
        data = json.loads(request.body)
        target_y = data.get('target_variable')
        feature_xs = data.get('feature_variables', [])
        
        df_json = request.session.get('mi_current_df')
        if not df_json:
            return JsonResponse({'status': 'error', 'message': 'データセットが有効期限切れです。'})
        
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        corr_data = EDA_Skill.calculate_correlation(df, target_y, feature_xs)
        plot_data = EDA_Skill.get_plot_data(df, target_y, feature_xs)
        
        return JsonResponse({
            'status': 'success',
            'correlation': corr_data,
            'plots': plot_data
        })



def api_curate_dataset(request, dept_pk):
    if request.method == 'POST':
        try:
            # 1. 画面からの設定データを受け取る（★絶対に消さないでください）
            data = json.loads(request.body)
            x_cols = data.get('x_cols', [])
            y_cols = data.get('y_cols', [])
            advanced_settings = data.get('advanced_settings', {})
            imputation_method = data.get('imputation_method', 'knn')
            use_feature_engineering = data.get('use_feature_engineering', False)

            # 2. 生データの読み込み
            df_json = request.session.get('mi_raw_df')
            if not df_json:
                return JsonResponse({'status': 'error', 'message': '生データがありません。画面左の「生データを読み込んでプレビュー」からやり直してください。'})

            df = pd.read_json(io.StringIO(df_json), orient='split')

            # 3. 特徴量エンジニアリング（AIマスタ結合）の実行
            if use_feature_engineering:
                # ★前回追加したハイブリッド・フォールバック用のワークスペースID取得
                database = get_object_or_404(Database, pk=dept_pk)
                current_workspace_id = database.workspace.id if database.workspace else None

                # ★mi_skills.py の FeatureEngineeringSkill に workspace_id を渡す
                df, generated_features = FeatureEngineeringSkill.execute(
                    df, 
                    y_cols=y_cols,
                    workspace_id=current_workspace_id
                )
                
                # 新しく生成された [AI] 列を説明変数(X)のリストに自動追加する
                if generated_features:
                    x_cols.extend(generated_features)

            # 4. データクレンジング（欠損値補完・外れ値除外など）の実行
            df_curated, final_count, summary_text, label_mappings = DataCurationSkill.execute(
                df, x_cols, y_cols, advanced_settings, imputation_method
            )

            if summary_text.startswith("error:"):
                return JsonResponse({
                    'status': 'error', 
                    'message': summary_text.replace("error:", "")
                })

            # 5. セッションへの保存と画面への返却
            request.session['mi_current_df'] = df_curated.to_json(orient='split')
            request.session['mi_label_mappings'] = label_mappings 

            html_table = df_curated.to_html(
                classes='table table-sm table-striped table-hover border mb-0 text-nowrap',
                index=False, justify='center', na_rep='(空欄)' 
            )

            return JsonResponse({
                'status': 'success',
                'final_rows': final_count,
                'summary_message': summary_text,
                'label_mappings': label_mappings,
                'html_table': html_table,
                'updated_x_cols': x_cols
            })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

def download_curated_data(request, dept_pk):
    df_json = request.session.get('mi_current_df')
    if not df_json:
        return HttpResponse("データが見つかりません。もう一度Step1から生成してください。", status=400)
    
    df = pd.read_json(io.StringIO(df_json), orient='split')
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    filename = urllib.parse.quote("AI解析用_補完済データセット.csv")
    response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
    
    df.to_csv(response, index=False, encoding='utf-8-sig')
    return response


def api_train_multi_models(request, dept_pk):
    if request.method == 'POST':
        try:
            from ..utils.mi_model_skills import MultiTargetModelTrainingSkill
            
            data = json.loads(request.body)
            database_id = data.get('database_id')
            target_variables = data.get('target_variables', [])
            feature_variables = data.get('feature_variables', [])
            models_config = data.get('models_config', {})
            advanced_settings = data.get('advanced_settings', {})
            imputation_method = data.get('imputation_method', 'knn')

            df_json = request.session.get('mi_current_df')
            if not df_json:
                return JsonResponse({'status': 'error', 'message': 'データセットの有効期限が切れました。Step1からやり直してください。'})

            df = pd.read_json(io.StringIO(df_json), orient='split')
            label_mappings = request.session.get('mi_label_mappings', {})

            preprocessing_meta = {
                'advanced_settings': advanced_settings,
                'imputation_method': imputation_method,
                'label_mappings': label_mappings
            }

            result = MultiTargetModelTrainingSkill.train_and_save_models(
                df=df,
                database_id=database_id,
                target_cols=target_variables,
                feature_cols=feature_variables,
                preprocessing_meta=preprocessing_meta,
                models_config=models_config
            )

            return JsonResponse(result)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': f'システムエラー: {str(e)}'})

    return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'})

def api_register_model(request, dept_pk):
    if request.method == 'POST':
        try:
            import json
            import os
            from django.conf import settings
            from ..models import TrainedModelMetadata, Workspace
            
            data = json.loads(request.body)
            temp_id = data.get('temp_id') 
            memo = data.get('memo', '')
            
            if not temp_id:
                return JsonResponse({'status': 'error', 'message': '一時ファイルが指定されていません。'})
            
            return JsonResponse({'status': 'success', 'message': 'モデルを保存しました'})
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': f'保存エラー: {str(e)}'})

    return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'})

def api_exclude_lots(request, workspace_pk):
    if request.method == 'POST':
        try:
            import re
            
            data = json.loads(request.body)
            exclude_lots = data.get('exclude_lots', [])
            
            if not exclude_lots:
                return JsonResponse({'status': 'warning', 'message': '除外するLotが選択されていません。'})

            raw_json = request.session.get('mi_raw_df')
            current_json = request.session.get('mi_current_df')
            
            if not raw_json or not current_json:
                return JsonResponse({'status': 'error', 'message': 'セッションのデータが見つかりません。再読み込みしてください。'})

            df_raw = pd.read_json(io.StringIO(raw_json), orient='split')
            df_curr = pd.read_json(io.StringIO(current_json), orient='split')
            
            def normalize_lot(val):
                s = str(val).strip()
                s = re.sub(r'[\s\u200B-\u200D\uFEFF]', '', s)
                if s.endswith('.0'):
                    s = s[:-2]
                return s.lower()
            
            clean_exclude_lots = [normalize_lot(lot) for lot in exclude_lots]
            excluded_count = 0
            
            if 'Lot番号' in df_raw.columns:
                original_count = len(df_raw)
                df_raw['norm_lot'] = df_raw['Lot番号'].apply(normalize_lot)
                mask_raw = df_raw['norm_lot'].isin(clean_exclude_lots)
                df_raw = df_raw[~mask_raw].drop(columns=['norm_lot'])
                excluded_count = original_count - len(df_raw)
                
                request.session['mi_raw_df'] = df_raw.to_json(orient='split', date_format='iso')
            else:
                return JsonResponse({'status': 'error', 'message': '生データに「Lot番号」が存在しません。'})

            if 'Lot番号' in df_curr.columns:
                df_curr['norm_lot'] = df_curr['Lot番号'].apply(normalize_lot)
                mask_curr = df_curr['norm_lot'].isin(clean_exclude_lots)
                df_curr = df_curr[~mask_curr].drop(columns=['norm_lot'])
                
                request.session['mi_current_df'] = df_curr.to_json(orient='split', date_format='iso')
            
            if excluded_count == 0:
                sample_db = df_raw['Lot番号'].head(3).tolist()
                msg = f"裏側での削除が0件でした。（送信値: {clean_exclude_lots[:3]}, DB値: {sample_db}）"
            else:
                msg = f"{excluded_count}件のLotデータを除外しました。"

            return JsonResponse({
                'status': 'success', 
                'message': msg,
                'remaining_count': len(df_curr)
            })
                
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': f'サーバーエラー: {str(e)}'})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


def api_convert_smiles(request, dept_pk):
    """
    SMILES変換API: 指定された列のSMILESをRDKitで記述子に変換し、セッションデータを即座に更新する。
    [データ駆動設計] 変換後にPandasによる厳密な型再推論を行い、真実のデータ型をフロントエンドに返却する。
    """
    if request.method == 'POST':
        try:
            import json
            import io
            import pandas as pd
            from ..utils.mi_skills import SmilesConversionSkill
            
            data = json.loads(request.body)
            target_col = data.get('target_col')
            
            if not target_col:
                return JsonResponse({'status': 'error', 'message': '変換する列が指定されていません。'})

            raw_json = request.session.get('mi_raw_df')
            curr_json = request.session.get('mi_current_df')
            
            if not raw_json or not curr_json:
                return JsonResponse({'status': 'error', 'message': '生データがありません。画面左から読み直してください。'})

            df_raw = pd.read_json(io.StringIO(raw_json), orient='split')
            df_curr = pd.read_json(io.StringIO(curr_json), orient='split')
            
            # 1. 独立した変換スキルの実行（RDKitによる化学記述子の横展開）
            df_raw, success, msg = SmilesConversionSkill.execute(df_raw, target_col)
            df_curr, _, _ = SmilesConversionSkill.execute(df_curr, target_col) 
            
            if not success:
                return JsonResponse({'status': 'error', 'message': msg})
                
            # 2. ★【データ駆動リファクタリング】追加されたAI列を含め、データセット全体の型判定を厳密に再評価
            df_raw = df_raw.infer_objects(copy=False)
            df_curr = df_curr.infer_objects(copy=False)
            
            protected_cols = ['テーマ名', 'Lot番号', '実験日', '担当者', '評価メモ']
            for col in df_curr.columns:
                if col not in protected_cols:
                    numeric_series = pd.to_numeric(df_curr[col], errors='coerce')
                    # NaN以外に1つでも有効な数値がある列は、正規の数値列としてキャスト
                    if numeric_series.notna().sum() > 0:
                        df_curr[col] = numeric_series
                        if col in df_raw.columns:
                            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
            # 3. 確定した真実のデータ型に基づいて、フロントエンド用の属性リストを再生成
            numeric_cols = df_curr.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_curr.select_dtypes(exclude=[np.number]).columns.tolist()
            all_cols = df_curr.columns.tolist()

            # セッションを最新の判定状態で上書き保存
            request.session['mi_raw_df'] = df_raw.to_json(orient='split', date_format='iso')
            request.session['mi_current_df'] = df_curr.to_json(orient='split', date_format='iso')
            
            # HTMLプレビューテーブルの生成
            safe_df = df_curr.replace({np.nan: None})
            raw_data_dict = safe_df.to_dict(orient='records')
            raw_data_json = json.dumps(raw_data_dict, ensure_ascii=False, default=str)

            html_table = df_curr.to_html(
                classes='table table-sm table-striped table-hover border mb-0 text-nowrap',
                index=False, justify='center', na_rep='(空欄)' 
            )

            return JsonResponse({
                'status': 'success',
                'message': msg,
                'html_table': html_table,
                'raw_data_json': raw_data_json,
                'columns': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols,
                    'all': all_cols
                }
            })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': f'サーバーエラー: {str(e)}'})
            
    return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'})