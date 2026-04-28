from django.views.generic import TemplateView
from django.shortcuts import get_object_or_404
import json
import io
from ..models import Workspace, Database
from django.http import JsonResponse
from ..models import Database, DatabaseItemDefinition, LotData
from ..utils.mi_skills import DatasetIntegrationSkill, SmartImputationSkill, EDA_Skill
import pandas as pd
import numpy as np

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
            # ★追加：万が一experimental_dataが空(None)でもエラーにならない安全対策
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
    対象部署のデータベース（テーマ）一覧を取得し、Phase 1（データ準備）の初期状態を構築する。
    """
    template_name = 'mi_analysis/base_analysis.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # urls.py で定義した dept_pk を使って部署情報を取得
        dept_pk = self.kwargs.get('dept_pk')
        workspace = get_object_or_404(Workspace, pk=dept_pk)
        context['workspace'] = workspace
        
        # この部署に紐づく実験テーマ（データベース）を取得
        databases = Database.objects.filter(workspace=workspace).order_by('-created_at')
        context['databases'] = databases
        
        return context



def api_generate_mi_dataset(request, dept_pk):
    """
    SkillNet API: フロントエンドからの非同期リクエストを受け取り、
    データ統合とスマート補完を実行してHTMLテーブルを返すエンドポイント。
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            database_ids = data.get('database_ids', [])
            imputation_method = data.get('imputation_method', 'knn')
            apply_outlier_filter = data.get('apply_outlier_filter', False)

            if not database_ids:
                return JsonResponse({'status': 'error', 'message': 'データベースが選択されていません。'})

            # 1. DatasetIntegrationSkill (統合)
            df, raw_count = DatasetIntegrationSkill.execute(database_ids)
            if df.empty:
                return JsonResponse({'status': 'error', 'message': '選択されたデータベースに有効なデータが存在しません。'})

            # 2. SmartImputationSkill (前処理・補完)
            df_clean, final_count, summary = SmartImputationSkill.execute(
                df=df, method=imputation_method, apply_outlier_filter=apply_outlier_filter
            )
            
            # セッションに保存
            request.session['mi_current_df'] = df_clean.to_json(orient='split')
            
            # カラムリストの抽出
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

            # PandasのDataFrameをHTMLに変換
            html_table = df_clean.to_html(
                classes='table table-sm table-striped table-hover border mb-0 text-nowrap',
                index=False, justify='center', na_rep='(空欄)' 
            )

            # ＝＝＝ バラバラだった return を1つに統合 ＝＝＝
            return JsonResponse({
                'status': 'success',
                'html_table': html_table,
                'total_rows': final_count,
                'summary_message': summary,
                'columns': {
                    'numeric': numeric_cols,
                    'categorical': categorical_cols,
                    'all': df_clean.columns.tolist()
                }
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'システムエラーが発生しました: {str(e)}'})

    return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'})

# (api_get_eda_results はそのまま変更なしでOKです！)
def api_get_eda_results(request, dept_pk):
    """Phase 2: 可視化と相関計算を実行するAPI"""
    if request.method == 'POST':
        data = json.loads(request.body)
        target_y = data.get('target_variable')
        feature_xs = data.get('feature_variables', [])
        
        df_json = request.session.get('mi_current_df')
        if not df_json:
            return JsonResponse({'status': 'error', 'message': 'データセットが有効期限切れです。'})
        
        # ＝＝＝ ★修正2: 文字列をStringIOで包むことでFutureWarningを完全に解消 ＝＝＝
        df = pd.read_json(io.StringIO(df_json), orient='split')
        
        corr_data = EDA_Skill.calculate_correlation(df, target_y, feature_xs)
        plot_data = EDA_Skill.get_plot_data(df, target_y, feature_xs)
        
        return JsonResponse({
            'status': 'success',
            'correlation': corr_data,
            'plots': plot_data
        })






def api_curate_dataset(request, dept_pk):
    """Phase 1: ユーザーが定義したスキーマに従ってデータを浄化するAPI"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            x_cols = data.get('x_cols', [])
            y_cols = data.get('y_cols', [])
            type_overrides = data.get('type_overrides', {})
            min_max_rules = data.get('min_max_rules', [])
            imputation_method = data.get('imputation_method', 'knn')

            df_json = request.session.get('mi_current_df')
            if not df_json:
                return JsonResponse({'status': 'error', 'message': '生データがありません。Step1からやり直してください。'})

            df = pd.read_json(io.StringIO(df_json), orient='split')

            # スキルの実行
            from ..utils.mi_skills import DataCurationSkill
            df_curated, final_count, summary_text, label_mappings = DataCurationSkill.execute(
                df, x_cols, y_cols, type_overrides, min_max_rules, imputation_method
            )

            # 解析用データをセッションに上書き保存 (以降のPhaseはこれを使う)
            request.session['mi_current_df'] = df_curated.to_json(orient='split')
            request.session['mi_label_mappings'] = label_mappings # ★この1行を追加！

            html_table = df_curated.to_html(
                classes='table table-sm table-striped table-hover border mb-0 text-nowrap',
                index=False, justify='center', na_rep='(空欄)' 
            )

            return JsonResponse({
                'status': 'success',
                'final_rows': final_count,
                'summary_message': summary_text,
                'label_mappings': label_mappings,
                'html_table': html_table  # ★この1行を追加！
            })
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})

# ＝＝＝ 2. ファイルの一番下に新規追加：CSVダウンロード用ビュー ＝＝＝
import urllib.parse
from django.http import HttpResponse

def download_curated_data(request, dept_pk):
    """浄化・補完済みのデータセットをCSVとしてダウンロードする"""
    df_json = request.session.get('mi_current_df')
    if not df_json:
        return HttpResponse("データが見つかりません。もう一度Step1から生成してください。", status=400)
    
    # セッションからDataFrameを復元
    df = pd.read_json(io.StringIO(df_json), orient='split')
    
   # ★修正：cp932ではなく「utf-8-sig (BOM付きUTF-8)」を使用する
    response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
    filename = urllib.parse.quote("AI解析用_補完済データセット.csv")
    response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
    
    # Pandasの機能を使って直接レスポンスにCSVを書き込む
    df.to_csv(response, index=False, encoding='utf-8-sig')
    return response