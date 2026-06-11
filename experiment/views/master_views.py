from django.views.generic import ListView, View
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
import json
import csv 
import urllib.parse 
import pandas as pd
import numpy as np
import datetime
from django.urls import reverse
import re
from django.db.models import Q
from ..models import Material, MaterialValue, MaterialPropertyDefinition, MasterDataPermission

class MasterManagePermissionMixin(UserPassesTestMixin):
    """ マスター管理者のみアクセスを許可するMixin """
    def test_func(self):
        return MasterDataPermission.can_manage(self.request.user)

# --- 1. マスターデータ一覧画面 ---
class MaterialMasterListView(LoginRequiredMixin, ListView):
    model = Material
    template_name = 'experiment/material_master_list.html'
    context_object_name = 'materials'
    paginate_by = 50

    def get_queryset(self):
        # 現在のユーザーが選択しているワークスペースIDを取得
        workspace_id = self.request.session.get('workspace_id')
        
        # 1. 物質のベースクエリ作成（全社共通マスター ＋ 自分のワークスペースマスター）
        queryset = super().get_queryset().filter(
            Q(workspace__isnull=True) | Q(workspace_id=workspace_id)
        ).prefetch_related('values__definition').order_by('-updated_at')
        
        # 2. 検索キーワードの判定と絞り込み（デッドコードを解消し、確実にここに到達させます）
        query = self.request.GET.get('q')
        if query:
            # IDに一致するか、または紐づくMaterialValueのいずれかにキーワードが含まれているかを検索
            queryset = queryset.filter(
                Q(id__icontains=query) |
                Q(values__value__icontains=query)
            ).distinct() # 重複を防ぐためにdistinct()を適用
            
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['can_manage_master'] = MasterDataPermission.can_manage(self.request.user)
        
        # 検索キーワードをテンプレートに返し、検索窓に文字を残す
        context['search_query'] = self.request.GET.get('q', '')
        
        # 現在のワークスペースIDを取得
        workspace_id = self.request.session.get('workspace_id')
        
        # ★修正: 列として表示する項目定義（ヘッダー）も、全社共通 ＋ 自ワークスペースのものだけに限定。
        # これにより、他部署が追加した特殊なオリジナル列が画面に侵入してグリッドを破壊するのを防ぎます。
        properties = MaterialPropertyDefinition.objects.filter(
            Q(workspace__isnull=True) | Q(workspace_id=workspace_id)
        ).order_by('order')
        context['properties'] = properties
        
        # 既存のHTMLテンプレートがそのまま綺麗に並べられるよう、定義順に値のリストを再構築
        for material in context['materials']:
            val_dict = {v.definition.property_key: v.value for v in material.values.all()}
            # 該当項目に値がない場合は '-' をデフォルト表示
            material.ordered_values = [val_dict.get(prop.property_key, '-') for prop in properties]
            
        return context

# --- 2. 項目設定画面 (Lotデータのsettingsと完全同一の動き) ---
class MaterialMasterSettingsView(LoginRequiredMixin, MasterManagePermissionMixin, View):
    template_name = 'experiment/material_master_settings.html'

    def get(self, request):
        properties = MaterialPropertyDefinition.objects.all().order_by('order')
        return render(request, self.template_name, {'properties': properties})

    def post(self, request):
        action = request.POST.get('action')
        try:
            if action == 'add':
                key = request.POST.get('property_key').strip()
                dtype = request.POST.get('data_type')
                
                # ★追加: 半角英数字とアンダースコアのみを許可するバリデーション
                if not re.match(r'^[a-zA-Z0-9_]+$', key):
                    return JsonResponse({
                        'status': 'error', 
                        'message': 'エラー: 項目名は半角英数字とアンダースコア(_)のみ使用可能です（日本語や空白は使用不可）。'
                    })

                if MaterialPropertyDefinition.objects.filter(property_key=key).exists():
                    return JsonResponse({'status': 'error', 'message': 'この項目キーは既に登録されています。'})
                
                MaterialPropertyDefinition.objects.create(
                    property_key=key, 
                    data_type=dtype,
                    order=MaterialPropertyDefinition.objects.count() + 1
                )
                return JsonResponse({'status': 'success'})

            elif action == 'edit':
                prop_id = request.POST.get('id')
                prop = MaterialPropertyDefinition.objects.get(id=prop_id)
                prop.data_type = request.POST.get('data_type')
                prop.save()
                return JsonResponse({'status': 'success'})

            elif action == 'delete':
                prop_id = request.POST.get('id')
                MaterialPropertyDefinition.objects.get(id=prop_id).delete()
                return JsonResponse({'status': 'success'})

            elif action == 'reorder':
                order_data = json.loads(request.POST.get('order_data', '[]'))
                for item in order_data:
                    MaterialPropertyDefinition.objects.filter(id=item['id']).update(order=item['order'])
                return JsonResponse({'status': 'success'})

            return JsonResponse({'status': 'error', 'message': '不正なアクションです。'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

# --- 3. 手動追加・編集画面 ---
class MaterialMasterFormView(LoginRequiredMixin, MasterManagePermissionMixin, View):
    def get(self, request, pk=None):
        material = get_object_or_404(Material, pk=pk) if pk else None
        properties = MaterialPropertyDefinition.objects.all().order_by('order')
        
        # 編集時は既存の値を辞書化してテンプレートへ渡す
        existing_values = {}
        if material:
            existing_values = {v.definition.property_key: v.value for v in material.values.all()}
            
        return render(request, 'experiment/material_master_form.html', {
            'object': material,
            'properties': properties,
            'existing_values': existing_values
        })

    def post(self, request, pk=None):
        # 新規作成か更新か
        if pk:
            material = get_object_or_404(Material, pk=pk)
            msg = 'マスターデータを更新しました。'
        else:
            material = Material.objects.create()
            msg = 'マスターデータを新規登録しました。'

        # フォームから送られた各項目の値を MaterialValue に保存
        for prop in MaterialPropertyDefinition.objects.all():
            val = request.POST.get(f'prop_{prop.property_key}')
            if val:
                MaterialValue.objects.update_or_create(
                    material=material,
                    definition=prop,
                    defaults={'value': val.strip()}
                )
            else:
                # 空欄で送信された場合は、既存の値を削除（クリア）する
                MaterialValue.objects.filter(material=material, definition=prop).delete()
        
        messages.success(request, msg)
        return redirect('master_list')

# --- 4. CSV/Excel 一括インポート画面 ---
class MaterialMasterImportView(LoginRequiredMixin, MasterManagePermissionMixin, View):
    template_name = 'experiment/material_master_import.html'

    def get(self, request):
        # 現在設定されている項目キーをテンプレートに渡す
        existing_keys = MaterialPropertyDefinition.objects.values_list('property_key', flat=True).order_by('order')
        return render(request, self.template_name, {
            'existing_keys': list(existing_keys)
        })

    def post(self, request):
        action = request.POST.get('action')
        
        if 'file' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'ファイルが選択されていません。'})

        uploaded_file = request.FILES['file']

        try:
            # 1. ファイルの読み込み
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='shift_jis')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError('CSVまたはExcelファイルを選択してください。')

            if df.empty:
                raise ValueError('ファイルにデータが含まれていません。')

            df = df.replace({np.nan: None})

            # 2. 項目定義の取得
            defs = {d.property_key: d for d in MaterialPropertyDefinition.objects.all()}
            existing_keys = set(defs.keys())
            
            # ★修正: 厳格なエラー（raise ValueError）を削除し、スルーする仕様に変更。

            # 3. プレビュー処理
            if action == 'preview':
                preview_data = []
                row1 = df.iloc[0].to_dict()

                has_japanese_header = False
                import re
                jp_pattern = re.compile(r'[^\x01-\x7E]') # マルチバイト文字検出

                for col in df.columns:
                    if 'Unnamed' in str(col): continue
                    val = row1.get(col, '')
                    if val is None or str(val).lower() == 'nan': val = '(空欄)'

                    col_str = str(col).strip()
                    is_invalid = bool(jp_pattern.search(col_str))
                    if is_invalid: has_japanese_header = True

                    # ★追加: 登録されていない列かどうかを判定
                    is_ignored = col_str not in existing_keys

                    preview_data.append({
                        'header': col_str, 
                        'value': str(val),
                        'is_invalid': is_invalid,
                        'is_ignored': is_ignored # JS側に渡す
                    })
                
                return JsonResponse({
                    'status': 'success', 
                    'preview': preview_data, 
                    'total_rows': len(df),
                    'has_japanese_header': has_japanese_header
                })

            # 4. 本番インポート実行処理
            elif action == 'execute':
                import_count = 0
                update_count = 0

                for index, row in df.iterrows():
                    cas_val = str(row.get('CAS_Number', '')).strip() if row.get('CAS_Number') else None
                    name_val = str(row.get('JapaneseName', '')).strip() if row.get('JapaneseName') else None

                    if not name_val and not cas_val:
                        continue # 空行はスキップ

                    # 既存データ（アンカー）の検索
                    target_material = None
                    if cas_val:
                        mv = MaterialValue.objects.filter(definition__property_key='CAS_Number', value=cas_val).first()
                        if mv: target_material = mv.material

                    if not target_material and name_val:
                        mv = MaterialValue.objects.filter(definition__property_key='JapaneseName', value=name_val).first()
                        if mv: target_material = mv.material

                    # 物質の親箱を作成または更新カウント
                    if target_material:
                        update_count += 1
                    else:
                        target_material = Material.objects.create()
                        import_count += 1

                    # 各列の値を MaterialValue テーブルに展開して保存
                    for col in df.columns:
                        if 'Unnamed' in str(col): continue
                        col_str = str(col).strip()
                        
                        # ★修正: 項目設定に無い列は単にスキップする
                        prop_def = defs.get(col_str)
                        if not prop_def: continue

                        val = row.get(col)
                        val_str = None

                        if val is not None and str(val).lower() != 'nan':
                            if hasattr(val, 'item'): val = val.item()
                            if isinstance(val, (datetime.date, datetime.datetime)): val = str(val)
                            else: val_str = str(val).strip()
                        else:
                            val_str = ""

                        if val_str:
                            MaterialValue.objects.update_or_create(
                                material=target_material,
                                definition=prop_def,
                                defaults={'value': val_str}
                            )
                        else:
                            MaterialValue.objects.filter(material=target_material, definition=prop_def).delete()

                messages.success(request, f'新規追加: {import_count}件、情報更新: {update_count}件 のデータを保存しました！（※未登録の列は無視されました）')
                return JsonResponse({'status': 'redirect', 'url': reverse('master_list')})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

# --- 5. Excelから項目名（ヘッダー）を自動抽出・登録する画面 ---
class MaterialItemImportView(LoginRequiredMixin, MasterManagePermissionMixin, View):
    template_name = 'experiment/material_master_item_import.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        if 'file' not in request.FILES:
            messages.error(request, 'ファイルが選択されていません。')
            return redirect('master_item_import')

        uploaded_file = request.FILES['file']

        try:
            # 1行目（ヘッダー）だけ読めば良いので、nrows=0で高速にカラム名だけ取得
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='shift_jis', nrows=0)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='utf-8', nrows=0)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, nrows=0)
            else:
                raise ValueError('CSVまたはExcelファイルを選択してください。')

            columns = [str(col).strip() for col in df.columns if 'Unnamed' not in str(col)]
            
            # バリデーションチェック（半角英数字とアンダースコアのみか）
            invalid_cols = [c for c in columns if not re.match(r'^[a-zA-Z0-9_]+$', c)]
            if invalid_cols:
                raise ValueError(f"エラー: 以下の列名に日本語やスペース、使用できない記号が含まれています。\n{', '.join(invalid_cols)}\n\n半角英数字とアンダースコア(_)のみに修正してから再度アップロードしてください。")

            # 既存の項目を取得
            existing_keys = set(MaterialPropertyDefinition.objects.values_list('property_key', flat=True))
            
            # 新規追加処理
            added_count = 0
            current_order = MaterialPropertyDefinition.objects.count()
            
            for col in columns:
                if col not in existing_keys:
                    current_order += 1
                    MaterialPropertyDefinition.objects.create(
                        property_key=col,
                        # ★修正: デフォルトのデータ型を 'text' から 'number'（数字）に変更
                        data_type='number', 
                        order=current_order
                    )
                    added_count += 1

            if added_count > 0:
                messages.success(request, f'{added_count} 個の新しい項目を自動抽出して登録しました！初期状態は「数字」に設定されています。必要に応じて文字列（JapaneseName等）に変更してください。')
            else:
                messages.info(request, 'ファイル内の項目はすべて既に登録されていました。追加された項目はありません。')

            return redirect('master_settings')

        except Exception as e:
            messages.error(request, str(e))
            return redirect('master_item_import')

# --- 6. 物性マスターのエクスポート ---
class MaterialExportView(LoginRequiredMixin, MasterManagePermissionMixin, View):
    def get(self, request, *args, **kwargs):
        today = datetime.datetime.now().strftime('%Y%m%d')
        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        filename = urllib.parse.quote(f"{today}_物性マスターデータ.csv")
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
        
        writer = csv.writer(response)
        
        # 項目定義を取得
        properties = MaterialPropertyDefinition.objects.all().order_by('order')
        headers = ['ID'] + [prop.property_key for prop in properties]
        writer.writerow(headers)
        
        # 全データを取得（prefetchでN+1回避）
        materials = Material.objects.prefetch_related('values__definition').order_by('-updated_at')
        
        for material in materials:
            val_dict = {v.definition.property_key: v.value for v in material.values.all()}
            row = [material.id] + [val_dict.get(prop.property_key, '') for prop in properties]
            writer.writerow(row)
            
        return response