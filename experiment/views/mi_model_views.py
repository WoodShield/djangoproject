import json
import io
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ..utils.mi_model_skills import MultiTargetModelTrainingSkill
from ..models import TrainedModelMetadata
from ..utils.mi_model_skills import MultiObjectiveInverseSkill
from ..utils.mi_skills import ModelTrainingSkill
import os 
import uuid
from django.utils import timezone 

def api_train_multi_models(request, dept_pk):
    """
    Phase 3: 複数目的変数の同時学習・保存を実行するAPI
    回帰・分類の自動判定と、Yellowbrickによる評価画像を生成して返す。
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '無効なリクエストです。'}, status=405)

    try:
        data = json.loads(request.body)
        database_id = data.get('database_id')
        target_y_list = data.get('target_variables', [])
        models_config = data.get('models_config', {})

        # ★修正1: フロントから送られてきた前処理の設定を受け取り、箱（辞書）にまとめる
        preprocessing_meta = {
            'type_overrides': data.get('type_overrides', {}),
            'min_max_rules': data.get('min_max_rules', []),
            'imputation_method': data.get('imputation_method', 'knn'),
            'saved_y_cols': target_y_list  
        }

        if not target_y_list:
            return JsonResponse({'status': 'error', 'message': '目的変数が選択されていません。'})

        df_json = request.session.get('mi_current_df')
        if not df_json:
            return JsonResponse({'status': 'error', 'message': 'データセットの有効期限切れです。'})
        
        df = pd.read_json(io.StringIO(df_json), orient='split')
        label_mappings = request.session.get('mi_label_mappings', {})

        final_details = []
        temp_models = request.session.get('temp_trained_models', {})

        for target in target_y_list:
            config = models_config.get(target, {})
            feature_cols = config.get('features', [])
            algorithm = config.get('algorithm', 'rf')
            hyperparams = config.get('hyperparams', {}) # ★追加: パラメータも受け取る

            result = ModelTrainingSkill.execute(
                df=df,
                target_col=target,
                feature_cols=feature_cols,
                algorithm=algorithm,
                label_mappings=label_mappings,
                hyperparams=hyperparams # ★追加: 万能スキルにパラメータを渡す
            )

            if result['status'] == 'success':
                result['target'] = target
                temp_id = str(uuid.uuid4())
                result['temp_id'] = temp_id

                # セッション（仮保存）にモデル情報を記憶
                temp_models[temp_id] = {
                    'database_id': database_id,
                    'target_variable': target,
                    'features_list': feature_cols,
                    'metrics': result['metrics'],
                    'task_type': result['task_type'],
                    'label_mappings': label_mappings,
                    'preprocessing_meta': preprocessing_meta, 
                    'temp_file_path': result.get('temp_file_path'),
                }
            
            final_details.append(result)

        request.session['temp_trained_models'] = temp_models

        return JsonResponse({
            'status': 'success',
            'details': final_details
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': f'サーバー内部エラー: {str(e)}'}, status=500)


def api_get_saved_models(request, dept_pk):
    """保存済みの学習済みモデル一覧をJSONで返すAPI"""
    try:
        database_id = request.GET.get('database_id')
        if not database_id:
            return JsonResponse({'status': 'error', 'message': 'テーマIDが指定されていません。'})

        models = TrainedModelMetadata.objects.filter(database_id=database_id).order_by('-created_at')
        
        model_list = []
        for m in models:
            local_time = timezone.localtime(m.created_at)
            model_list.append({
                'id': m.id,
                'target': m.target_variable,
                'metrics': m.metrics,
                'features': m.features_list,
                'preprocessing_meta': m.preprocessing_meta,
                'created_at': local_time.strftime('%Y/%m/%d %H:%M')
            })
            
        return JsonResponse({'status': 'success', 'models': model_list})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})

def api_optimize_recipe(request, dept_pk):
    """Phase 4: 保存済みモデルをロードして多目的最適化（逆解析）を実行するAPI"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_ids = data.get('model_ids', [])
            target_goals = data.get('target_goals', {})

            if not model_ids:
                return JsonResponse({'status': 'error', 'message': 'モデルが選択されていません。'})

            df_json = request.session.get('mi_current_df')
            df = None
            if df_json:
                df = pd.read_json(io.StringIO(df_json), orient='split')

            result = MultiObjectiveInverseSkill.run_multi_optimization(model_ids, target_goals, df)
            return JsonResponse(result)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)})
            
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


def api_register_model(request, dept_pk):
    """手動セーブボタンが押された時にDBに正式保存するAPI"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            temp_id = data.get('temp_id')
            memo = data.get('memo', '') 
            
            temp_models = request.session.get('temp_trained_models', {})
            m_data = temp_models.get(temp_id)
            
            if not m_data:
                return JsonResponse({'status': 'error', 'message': 'セッションが切れました。再度学習してください。'})
            
            # ★修正3: 万が一 'preprocessing_meta' が無い場合でもエラーで落ちないようにする安全装置
            if 'preprocessing_meta' not in m_data:
                m_data['preprocessing_meta'] = {}
                
            m_data['preprocessing_meta']['memo'] = memo
            
            from ..models import Database
            from django.core.files import File
            from django.conf import settings

            database = Database.objects.get(id=m_data['database_id'])
            
            metadata = TrainedModelMetadata.objects.create(
                database=database,
                target_variable=m_data['target_variable'],
                features_list=m_data['features_list'],
                preprocessing_meta=m_data['preprocessing_meta'],
                metrics=m_data['metrics']
            )
            temp_path_str = m_data.get('temp_file_path')
            if temp_path_str:
                full_temp_path = os.path.join(settings.MEDIA_ROOT, temp_path_str)
                if os.path.exists(full_temp_path):
                    with open(full_temp_path, 'rb') as f:
                        # 'model_final_xxxx.pkl' という名前でDBの FileField に保存
                        final_filename = os.path.basename(full_temp_path).replace('model_temp_', 'model_final_')
                        metadata.model_file.save(final_filename, File(f), save=False)
                    
                    # ストレージ節約のため、紐付けが完了した一時ファイルは削除
                    try:
                        os.remove(full_temp_path)
                    except OSError:
                        pass
           
            metadata.save()
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'status': 'error', 'message': str(e)})

def api_delete_model(request, dept_pk):
    """ユーザーがモデル管理画面からモデルを削除するAPI"""
    if request.method == 'POST':
        try:
            model_id = json.loads(request.body).get('model_id')
            m = TrainedModelMetadata.objects.get(id=model_id)
            
            if m.model_file and os.path.exists(m.model_file.path):
                os.remove(m.model_file.path)
            m.delete()
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})