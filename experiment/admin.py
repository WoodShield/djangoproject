import csv
from django.http import HttpResponse
from django.contrib import admin

from .models import Workspace, Database, DatabaseItemDefinition, LotData, TrainedModelMetadata, Organization, UserProfile
# =========================================================
# 汎用的なCSVエクスポート関数の定義
# =========================================================
def export_to_csv(modeladmin, request, queryset, filename, headers, row_func):
    """
    modelごとに異なる項目を出力するための汎用関数
    """
    response = HttpResponse(content_type='text/csv; charset=cp932')
    response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(headers)
    
    for obj in queryset:
        writer.writerow(row_func(obj))
    return response

# --- 各モデル専用のエクスポートアクション ---

@admin.action(description='選択したワークスペースをCSVで保存')
def export_workspaces_csv(modeladmin, request, queryset):
    headers = ['ID', '名前', '作成日時']
    row_func = lambda obj: [obj.id, obj.name, obj.created_at]
    return export_to_csv(modeladmin, request, queryset, "workspaces", headers, row_func)

@admin.action(description='選択したテーマ設定をCSVで保存')
def export_databases_csv(modeladmin, request, queryset):
    headers = ['ID', '名前', 'ワークスペース', '作成者', '異常検知設定', '更新日時']
    row_func = lambda obj: [
        obj.id, obj.name, obj.workspace.name if obj.workspace else '',
        obj.author, "ON" if obj.enable_anomaly_check else "OFF", obj.updated_at
    ]
    return export_to_csv(modeladmin, request, queryset, "databases", headers, row_func)

@admin.action(description='選択したLotデータをCSVで保存')
def export_lotdata_csv(modeladmin, request, queryset):
    headers = ['Lot番号', 'データベース名', '実験日', '担当者', 'メモ', '登録日時']
    row_func = lambda obj: [
        obj.lot_number, obj.database.name if obj.database else '',
        obj.recorded_date, obj.recorded_by, obj.evaluation_memo, obj.created_at
    ]
    return export_to_csv(modeladmin, request, queryset, "lot_data", headers, row_func)

@admin.action(description='選択した項目定義をCSVで保存')
def export_item_definitions_csv(modeladmin, request, queryset):
    headers = ['ID', 'テーマ名', '項目名', 'データの種類', '選択肢', '計算式', '異常検知対象', '入力表示', '並び順']
    row_func = lambda obj: [
        obj.id, obj.database.name if obj.database else '', obj.item_name,
        obj.get_data_type_display(), obj.choices_text, obj.calculation_formula,
        "ON" if obj.use_for_anomaly else "OFF", "表示" if obj.is_active else "非表示", obj.order
    ]
    return export_to_csv(modeladmin, request, queryset, "item_definitions", headers, row_func)

@admin.action(description='選択した学習済みモデルをCSVで保存')
def export_trained_models_csv(modeladmin, request, queryset):
    headers = ['ID', 'テーマ名', '目的変数', 'R2スコア', '作成日時']
    
    def row_func(obj):
        # json(metrics)の中からR2スコアを取り出して丸める
        r2 = obj.metrics.get('r2') if obj.metrics else ''
        if isinstance(r2, float):
            r2 = round(r2, 3)
            
        return [
            obj.id, 
            obj.database.name if obj.database else '', 
            obj.target_variable, 
            r2, 
            obj.created_at.strftime('%Y/%m/%d %H:%M') if obj.created_at else ''
        ]
        
    return export_to_csv(modeladmin, request, queryset, "trained_models", headers, row_func)


# =========================================================
# 管理画面への登録（各Adminクラスにアクションを紐づけ）
# =========================================================

@admin.register(Workspace)
class WorkspaceAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at')
    actions = [export_workspaces_csv] # アクション追加

@admin.register(Database)
class DatabaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'workspace', 'author', 'updated_at')
    list_filter = ('workspace', 'author')
    actions = [export_databases_csv] # アクション追加

@admin.register(LotData)
class LotDataAdmin(admin.ModelAdmin):
    list_display = ('lot_number', 'database', 'recorded_date', 'recorded_by')
    list_filter = ('database',)
    actions = [export_lotdata_csv] # アクション追加

@admin.register(DatabaseItemDefinition)
class DatabaseItemDefinitionAdmin(admin.ModelAdmin):
    list_display = ('item_name', 'database', 'data_type', 'is_active', 'order')
    list_filter = ('database', 'data_type', 'is_active')
    ordering = ('database', 'order')
    actions = [export_item_definitions_csv] # アクション追加

@admin.register(TrainedModelMetadata)
class TrainedModelMetadataAdmin(admin.ModelAdmin):
    list_display = ('database', 'target_variable', 'get_r2_score', 'created_at')
    list_filter = ('database',)
    search_fields = ('database__name', 'target_variable')
    readonly_fields = ('created_at',)
    
    # ★ ここに作成したCSVダウンロードアクションを追加！
    actions = [export_trained_models_csv]

    def get_r2_score(self, obj):
        """一覧画面にR2スコアを抽出して表示"""
        r2 = obj.metrics.get('r2') if obj.metrics else None
        if isinstance(r2, float):
            return round(r2, 3)
        return r2
    get_r2_score.short_description = 'R2スコア'

# =========================================================
# 組織・ユーザープロフィールの管理画面登録
# =========================================================

@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at')
    search_fields = ('name',)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'display_name', 'organization')
    list_filter = ('organization',)
    search_fields = ('user__username', 'display_name')