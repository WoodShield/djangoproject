from django.urls import path
from .views.workspace_views import WorkspaceListView, WorkspaceCreateView,WorkspaceUpdateView

from .views.database_views import ThemeListView, ThemeCreateView, ThemeSettingsView, DatabaseUpdateView
from .views.lot_views import LotListView, LotCreateView, LotUpdateView, BatchRecheckAnomalyView, LotComparisonView
from .views.import_export import LotDataImportView, ThemeItemImportView, ThemeExportView, LotDataExportView

# analysis_viewsからはPhase 1と2のAPIのみをインポート
from .views.analysis_views import LotAnalysisView, MLAnalysisView, MIAnalysisView, api_generate_mi_dataset, api_get_eda_results, api_curate_dataset, download_curated_data

# mi_model_viewsからPhase 3と4のAPIをまとめてインポート
from .views.mi_model_views import api_train_multi_models, api_get_saved_models, api_optimize_recipe,api_register_model, api_delete_model

from django.contrib.auth import views as auth_views 
from .views.auth_views import MyPageView, OrganizationCreateView, SignUpView



urlpatterns = [
    # --- ログイン・マイページ ---
    path('login/', auth_views.LoginView.as_view(template_name='experiment/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('mypage/', MyPageView.as_view(), name='mypage'),
    
    # 部署（組織）の新規登録URL
    path('organization/create/', OrganizationCreateView.as_view(), name='organization_create'),
    # 新規ユーザー登録URL
    path('signup/', SignUpView.as_view(), name='signup'),


    # --- 1階層目: ワークスペース（部署） ---
    path('', WorkspaceListView.as_view(), name='workspace_list'),
    path('dept/create/', WorkspaceCreateView.as_view(), name='workspace_create'),
    path('dept/<int:pk>/update/',WorkspaceUpdateView.as_view(), name='workspace_update'),

    # --- 2階層目: 特定のワークスペースに紐づくテーマ一覧 ---
    path('dept/<int:dept_pk>/databases/', ThemeListView.as_view(), name='database_list'),
    path('dept/<int:dept_pk>/databases/create/', ThemeCreateView.as_view(), name='database_create'),

    # 既存の MI解析エンジンの画面表示用
    path('dept/<int:dept_pk>/mi_analysis/', MIAnalysisView.as_view(), name='mi_analysis'),

    # Phase 1: 非同期通信用のAPIエンドポイント (データ生成)
    path('dept/<int:dept_pk>/api/dataset/generate/', api_generate_mi_dataset, name='api_generate_mi_dataset'),
    path('workspace/<int:dept_pk>/api/curate/', api_curate_dataset, name='api_curate_dataset'),
    path('workspace/<int:dept_pk>/api/curate/download/', download_curated_data, name='download_curated_data'),
    
    # Phase 2: EDA（可視化・相関）用のAPIエンドポイント
    path('dept/<int:dept_pk>/api/eda/', api_get_eda_results, name='api_get_eda_results'),

    # Phase 3: モデル学習用のAPIエンドポイント (古い単一モデル用を削除し、一括学習用のみ残す)
    path('dept/<int:dept_pk>/api/model/train_multi/', api_train_multi_models, name='api_train_multi_models'),
    
    # Phase 4: 逆解析・最適化用のAPIエンドポイント (モデル一覧取得APIを追加)
    path('dept/<int:dept_pk>/api/model/list/', api_get_saved_models, name='api_get_saved_models'),
    path('dept/<int:dept_pk>/api/model/optimize/', api_optimize_recipe, name='api_optimize_recipe'),

    # === 学習モデル管理 ===
    path('dept/<int:dept_pk>/api/model/register/', api_register_model, name='api_register_model'), 
    path('dept/<int:dept_pk>/api/model/delete/', api_delete_model, name='api_delete_model'), 

    # テーマ一覧のCSV出力
    path('databases/export/', ThemeExportView.as_view(), name='database_export'), 

    # --- 3階層目以降: テーマ詳細・Lotデータ（ワークスペースIDに依存させず独立） ---
    path('database/<int:pk>/lots/', LotListView.as_view(), name='lot_list'),
    path('database/<int:pk>/settings/', ThemeSettingsView.as_view(), name='database_settings'),
    path('database/<int:pk>/analysis/', LotAnalysisView.as_view(), name='lot_analysis'),
    path('database/<int:pk>/update/', DatabaseUpdateView.as_view(), name='database_update'),
    
    # 登録・編集・出力・インポート
    path('database/<int:pk>/lots/create/', LotCreateView.as_view(), name='lot_create'),
    path('database/<int:pk>/recheck_anomalies/', BatchRecheckAnomalyView.as_view(), name='batch_recheck_anomalies'),
    path('database/<int:database_pk>/lots/<int:pk>/update/', LotUpdateView.as_view(), name='lot_update'),
    path('database/<int:pk>/lots/compare/', LotComparisonView.as_view(), name='lot_compare'),
    path('database/<int:pk>/export-lots/', LotDataExportView.as_view(), name='lot_export'),
    path('database/<int:pk>/import-data/', LotDataImportView.as_view(), name='lot_import'),
    
    # 項目設定のExcelインポート
    path('database/<int:pk>/import-items/', ThemeItemImportView.as_view(), name='database_item_import'),
]