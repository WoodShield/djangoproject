from django.views.generic import ListView, CreateView, UpdateView
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from ..models import Database, LotData
from ..forms import LotDataForm
from ..services import ExperimentAIAnalyzer
from django.views import View
import os
from django.core.files.storage import default_storage
import re
import math


def calculate_auto_fields(database, data_dict):
    """内部名や全角記号の罠をすべて回避する最強の自動計算エンジン"""
    
    # ★改善1: data_typeの内部名に依存せず、「計算式(formula)が入力されている項目」をすべて対象にする！
    calc_fields = database.items.exclude(calculation_formula='').exclude(calculation_formula__isnull=True).order_by('order')
    
    for field in calc_fields:
        formula = field.calculation_formula
        if not formula: continue
        
        # ★改善2: 日本語入力でありがちな全角演算子（＋、－、＊、／）を、強制的に半角に自動変換する
        formula = formula.replace('＋', '+').replace('－', '-').replace('＊', '*').replace('／', '/')
        
        variables = re.findall(r'\{([^}]+)\}', formula)
        expr = formula
        try:
            for var in variables:
                val = data_dict.get(var, 0)
                if val == '' or val is None: val = 0
                else:
                    try: val = float(val)
                    except (ValueError, TypeError): val = 0
                expr = expr.replace(f'{{{var}}}', str(val))
                
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            expr = expr.replace('^', '**')
            
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            data_dict[field.item_name] = round(float(result), 5)
            
            # デバッグ用: 計算が成功したらコンソールに緑色っぽく表示
            print(f"✅ 計算成功 [{field.item_name}]: {result} (評価した式: {expr})")
            
        except Exception as e:
            # デバッグ用: 計算が失敗したらコンソールに赤裸々に表示
            print(f"❌ 計算エラー [{field.item_name}]: {e} (評価した式: {expr})")
            data_dict[field.item_name] = 0
            
    return data_dict

    
class LotListView(ListView):
    model = LotData
    template_name = 'experiment/lot_list.html'
    context_object_name = 'lots'

    def get_queryset(self):
        database_id = self.kwargs.get('pk')
        return LotData.objects.filter(database_id=database_id).order_by('-recorded_date')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        database = get_object_or_404(Database, pk=self.kwargs.get('pk'))
        context['database'] = database
        context['item_defs'] = database.items.filter(is_active=True).order_by('order')

        lots = context['lots']
        context['anomaly_lots'] = [lot for lot in lots if lot.ai_anomaly_msg and not lot.is_anomaly_acknowledged]
        
        # ＝＝＝ ★UI表示用のデータ整形スキル（画像ファイル名の抽出） ＝＝＝
        for lot in context['lots']:
            display_data = {}
            raw_data = lot.experimental_data or {}
            
            for key, value in raw_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str) and value[0].startswith('databases/'):
                    image_list = []
                    for path in value:
                        filename = path.split('/')[-1]
                        image_list.append({'path': path, 'name': filename})
                    display_data[key] = image_list
                else:
                    display_data[key] = value
                    
            lot.display_data = display_data
            
        return context


class LotCreateView(CreateView):
    model = LotData
    form_class = LotDataForm
    template_name = 'experiment/lot_form.html'

    def get_database(self):
        return get_object_or_404(Database, pk=self.kwargs.get('pk'))

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['database'] = self.get_database()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['database'] = self.get_database()
        return context

    def form_valid(self, form):
        form.instance.database = self.get_database()
        lot_num = form.cleaned_data.get('lot_number')
        
        experimental_data = {}
        
        # 1. 画像ファイルの保存処理
        for field_name in self.request.FILES:
            if field_name.startswith('dynamic_'):
                real_item_name = field_name.replace('dynamic_', '')
                files = self.request.FILES.getlist(field_name)
                
                saved_paths = []
                for i, f in enumerate(files):
                    ext = os.path.splitext(f.name)[1]
                    original_name_no_ext = os.path.splitext(f.name)[0]
                    new_filename = f"databases/{self.get_database().id}/{lot_num}_{real_item_name}_{i+1}({original_name_no_ext}){ext}"
                    path = default_storage.save(new_filename, f)
                    saved_paths.append(path)
                
                experimental_data[real_item_name] = saved_paths

        # 2. テキスト・数値データの保存（画像以外の項目だけを処理）
        for field_name, value in form.cleaned_data.items():
            if field_name.startswith('dynamic_'):
                real_item_name = field_name.replace('dynamic_', '')
                if field_name not in self.request.FILES:
                    experimental_data[real_item_name] = value

        experimental_data = calculate_auto_fields(self.get_database(), experimental_data)
        
        form.instance.is_anomaly_acknowledged = False
        form.instance.experimental_data = experimental_data
        
        database = self.get_database()
        anomaly_msg = None
        if database.enable_anomaly_check:
            anomaly_msg = ExperimentAIAnalyzer.check_anomaly(database, experimental_data, None)

        form.instance.ai_anomaly_msg = anomaly_msg
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('lot_list', kwargs={'pk': self.get_database().pk})


class LotUpdateView(UpdateView):
    model = LotData
    form_class = LotDataForm
    template_name = 'experiment/lot_form.html'

    def get_database(self):
        return get_object_or_404(Database, pk=self.kwargs.get('database_pk'))

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['database'] = self.get_database()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['database'] = self.get_database()
        return context

    def form_valid(self, form):
        lot_num = form.cleaned_data.get('lot_number')
        experimental_data = self.object.experimental_data or {}

        # ＝＝＝ ★既存画像の削除処理（JSONと物理ファイルの両方から消す） ＝＝＝
        for post_key in self.request.POST:
            if post_key.startswith('delete_image_dynamic_'):
                real_item_name = post_key.replace('delete_image_dynamic_', '')
                paths_to_delete = self.request.POST.getlist(post_key)
                current_images = experimental_data.get(real_item_name, [])
                
                for path in paths_to_delete:
                    if path in current_images:
                        current_images.remove(path) # JSONのリストから削除
                        if default_storage.exists(path):
                            default_storage.delete(path) # サーバーの画像フォルダからも完全に削除
                
                experimental_data[real_item_name] = current_images
        # ＝＝＝ ここまで ＝＝＝
        
        # ＝＝＝ ★画像ファイルの保存処理（追加モード） ＝＝＝
        for field_name in self.request.FILES:
            if field_name.startswith('dynamic_'):
                real_item_name = field_name.replace('dynamic_', '')
                files = self.request.FILES.getlist(field_name)
                
                new_saved_paths = []
                for i, f in enumerate(files):
                    ext = os.path.splitext(f.name)[1]
                    original_name_no_ext = os.path.splitext(f.name)[0]
                    
                    # 保存済みの枚数を確認して、連番が重ならないようにする
                    current_images = experimental_data.get(real_item_name, [])
                    next_index = len(current_images) + i + 1
                    
                    new_filename = f"databases/{self.get_database().id}/{lot_num}_{real_item_name}_{next_index}({original_name_no_ext}){ext}"
                    path = default_storage.save(new_filename, f)
                    new_saved_paths.append(path)
                
                # ＝＝＝ ★ここがポイント：上書きではなく「既存リストに追加」 ＝＝＝
                if real_item_name in experimental_data and isinstance(experimental_data[real_item_name], list):
                    experimental_data[real_item_name].extend(new_saved_paths)
                else:
                    experimental_data[real_item_name] = new_saved_paths

        # テキスト・数値データの保存（画像以外の項目を処理）
        for field_name, value in form.cleaned_data.items():
            if field_name.startswith('dynamic_'):
                real_item_name = field_name.replace('dynamic_', '')
                
                # 画像フィールド以外、または画像がアップロードされていない場合に値を更新
                if field_name not in self.request.FILES:
                    existing_val = experimental_data.get(real_item_name)
                    # 既存が画像のリストなら、何もしない（＝維持する）
                    if isinstance(existing_val, list) and len(existing_val) > 0 and isinstance(existing_val[0], str) and existing_val[0].startswith('databases/'):
                        pass 
                    else:
                        experimental_data[real_item_name] = value
        experimental_data = calculate_auto_fields(self.get_database(), experimental_data)
                
        form.instance.experimental_data = experimental_data
        
        database = self.get_database()
        anomaly_msg = None
        if database.enable_anomaly_check:
            anomaly_msg = ExperimentAIAnalyzer.check_anomaly(database, experimental_data, self.object.id)

        form.instance.ai_anomaly_msg = anomaly_msg
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('lot_list', kwargs={'pk': self.get_database().pk})

class BatchRecheckAnomalyView(View):
    """
    最新のAIで過去の全データを一括再審査するビュー
    """
    def post(self, request, pk):
        database = get_object_or_404(Database, pk=pk)
        lots = database.lots.all()
        
        updated_count = 0
        for lot in lots:
            if lot.experimental_data:
                lot.is_anomaly_acknowledged = False
                
                
                # ＝＝＝ AI機能のON/OFF制御（一括再審査） ＝＝＝
                anomaly_msg = None
                if database.enable_anomaly_check:
                    anomaly_msg = ExperimentAIAnalyzer.check_anomaly(database, lot.experimental_data, lot.id)
                    
                
                
                lot.ai_anomaly_msg = anomaly_msg
                
                lot.save()
                
                updated_count += 1

        messages.success(request, f"{updated_count}件のデータを最新のAI基準で一括再審査しました！")
        return redirect('lot_list', pk=pk)



class LotComparisonView(ListView):
    """
    複数のLotを横並びで比較する専用ダッシュボード
    """
    model = LotData
    template_name = 'experiment/lot_comparison.html'
    context_object_name = 'lots'

    def get_queryset(self):
        database_id = self.kwargs.get('pk')
        # URLのパラメータ（?ids=181,182...）からIDのリストを取得
        ids_str = self.request.GET.get('ids', '')
        if not ids_str:
            return LotData.objects.none()
            
        try:
            lot_ids = [int(id_str) for id_str in ids_str.split(',')]
            # 渡されたIDに合致するデータだけを取得（Lot番号順に並べ替え）
            return LotData.objects.filter(database_id=database_id, pk__in=lot_ids).order_by('lot_number')
        except ValueError:
            return LotData.objects.none()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        database = get_object_or_404(Database, pk=self.kwargs.get('pk'))
        context['database'] = database
        context['item_defs'] = database.items.filter(is_active=True).order_by('order')

        # ＝＝＝ 比較画面用に画像データを整形 ＝＝＝
        for lot in context['lots']:
            display_data = {}
            raw_data = lot.experimental_data or {}
            
            for key, value in raw_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str) and value[0].startswith('databases/'):
                    image_list = []
                    for path in value:
                        filename = path.split('/')[-1]
                        image_list.append({'path': path, 'name': filename})
                    display_data[key] = image_list
                else:
                    display_data[key] = value
                    
            lot.display_data = display_data
            
        return context

