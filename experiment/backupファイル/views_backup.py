from django.views.generic import ListView, CreateView, UpdateView 
from django.urls import reverse_lazy
from django.shortcuts import get_object_or_404, redirect
from django.forms.models import inlineformset_factory, BaseInlineFormSet
from django.contrib import messages
from .models import ExperimentTheme, LotData, ThemeItemDefinition 
from .forms import LotDataForm, DataImportForm, ThemeItemDefinitionForm
import pandas as pd
from django.utils.dateparse import parse_date
from datetime import datetime
from django.core.exceptions import ValidationError
import csv
import urllib.parse
from django.http import HttpResponse
from django.views import View
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from django.views.generic import TemplateView
import json

class ThemeListView(ListView):
    model = ExperimentTheme
    # 作成するHTMLファイルを指定
    template_name = 'experiment/theme_list.html'
    # HTML側でデータを受け取る時の変数名
    context_object_name = 'themes'
    # 追加：更新日の降順（新しい順）で並び替え
    def get_queryset(self):
        return ExperimentTheme.objects.all().order_by('-updated_at')

class ThemeCreateView(CreateView):
    model = ExperimentTheme
    # 画面に入力欄として表示させたい項目を指定
    fields = ['name', 'author', 'enable_anomaly_check', 'enable_similarity_search']
    template_name = 'experiment/theme_form.html'
    # 登録完了後に戻るページ（トップページ）のURL名
    success_url = reverse_lazy('theme_list')

class LotListView(ListView):
    model = LotData
    template_name = 'experiment/lot_list.html'
    context_object_name = 'lots'

    # URLに含まれるテーマID(pk)を使って、そのテーマのLotデータだけに絞り込む
    def get_queryset(self):
        theme_id = self.kwargs.get('pk')
        return LotData.objects.filter(theme_id=theme_id).order_by('-recorded_date')

    # 画面側でテーマの名前や「設定ボタン」を表示できるように、テーマ自体の情報も渡す
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        theme = get_object_or_404(ExperimentTheme, pk=self.kwargs.get('pk'))
        context['theme'] = theme
        
        # 設定された有効な項目名を取得（これを目次にする）
        context['item_defs'] = theme.items.filter(is_active=True).order_by('order')
        
        # テンプレート側で扱いやすいように、各LotのJSONデータを加工
        # 各lotに .display_data という属性を持たせて、項目名で値を取り出せるようにする
        for lot in context['lots']:
            lot.display_data = lot.experimental_data or {}
            
        return context


# ① フォーム全体をチェックして、ダブりがないか確認するカスタムルール
class BaseThemeItemFormSet(BaseInlineFormSet):
    def clean(self):
        super().clean()
        if any(self.errors):
            return # すでに個別の入力エラーがある場合は一旦ストップ
        
        item_names = []
        for form in self.forms:
            # 削除にチェックが入っている行は、これから消えるのでダブりチェックから除外
            if self.can_delete and self._should_delete_form(form):
                continue
            
            name = form.cleaned_data.get('item_name')
            if name:
                # すでにリストに同じ名前が存在していれば、エラーを発生させる！
                if name in item_names:
                    raise ValidationError(f"⚠️ エラー：項目名「{name}」が重複して入力されています。別の名前に変更してください。")
                item_names.append(name)

# ② 魔法のフォームセットに、今作ったカスタムルール（formset=）を適用する
ItemFormSet = inlineformset_factory(
    ExperimentTheme, ThemeItemDefinition,
    form=ThemeItemDefinitionForm,
    formset=BaseThemeItemFormSet,  # ← ★この1行を追加してください！★
    extra=1, 
    can_delete=True
)

class ThemeSettingsView(UpdateView):
    model = ExperimentTheme
    # ご要望通り、ここでもAIのON/OFFを変更できるようにします
    fields = ['enable_anomaly_check', 'enable_similarity_search']
    template_name = 'experiment/theme_settings.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.POST:
            context['item_formset'] = ItemFormSet(self.request.POST, instance=self.object)
        else:
            context['item_formset'] = ItemFormSet(instance=self.object)
        return context

    def form_valid(self, form):
        context = self.get_context_data()
        item_formset = context['item_formset']
        if item_formset.is_valid():
            self.object = form.save()
            item_formset.instance = self.object
            item_formset.save()
            messages.success(self.request, '設定を保存しました。')
            return super().form_valid(form)
        else:
            # ★追加：エラーがある場合は赤い警告メッセージを出す
            messages.error(self.request, '入力内容にエラーがあります。')
            return self.render_to_response(self.get_context_data(form=form))

   
    def get_success_url(self):
        # 「一覧に戻る」ボタンが押された場合
        if 'save_and_return' in self.request.POST:
            return reverse_lazy('lot_list', kwargs={'pk': self.object.pk})
        # それ以外（追加して保存ボタン）の場合は、今の設定画面にとどまる
        return reverse_lazy('theme_settings', kwargs={'pk': self.object.pk})

class LotCreateView(CreateView):
    model = LotData
    form_class = LotDataForm  # 先ほど作った魔法のフォームを指定
    template_name = 'experiment/lot_form.html'

    # URLに含まれるテーマIDを使って、どのテーマのLotデータかを特定する
    def get_theme(self):
        theme_id = self.kwargs.get('pk')
        return get_object_or_404(ExperimentTheme, pk=theme_id)

    # フォームにテーマ情報を渡す（これで forms.py が項目を生成できる）
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['theme'] = self.get_theme()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['theme'] = self.get_theme()
        return context

    # 保存する直前に、動的に作られたフィールドの値を JSON(experimental_data) にまとめる
    def form_valid(self, form):
        form.instance.theme = self.get_theme() # Lotデータにテーマを紐づける
        
        # dynamic_ から始まるフィールドの入力値を抽出して辞書（JSONの元）にする
        experimental_data = {}
        for field_name, value in form.cleaned_data.items():
            if field_name.startswith('dynamic_'):
                # 辞書のキーには実際の項目名（dynamic_以降の文字）を使う
                real_item_name = field_name.replace('dynamic_', '')
                experimental_data[real_item_name] = value

        # JSONFieldにセット！
        form.instance.experimental_data = experimental_data
        
        return super().form_valid(form)

    def get_success_url(self):
        # 登録後はLot一覧画面に戻る
        return reverse_lazy('lot_list', kwargs={'pk': self.get_theme().pk})

class LotDataImportView(UpdateView):
    template_name = 'experiment/lot_import.html'
    
    def get(self, request, pk):
        theme = get_object_or_404(ExperimentTheme, pk=pk)
        form = DataImportForm()
        return self.render_to_response({'form': form, 'theme': theme})

    def post(self, request, pk):
        theme = get_object_or_404(ExperimentTheme, pk=pk)
        form = DataImportForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['file']
            try:
                # 拡張子でCSVかExcelかを判定して読み込む
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='shift_jis')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    messages.error(request, 'CSVまたはExcelファイルを選択してください。')
                    return self.render_to_response({'form': form, 'theme': theme})

                # NaN（空のセル）を None に変換
                df = df.where(pd.notnull(df), None)

                # 設定画面で作った項目の一覧を取得
                item_defs = ThemeItemDefinition.objects.filter(theme=theme, is_active=True)
                item_names = [item.item_name for item in item_defs]

                import_count = 0
                for index, row in df.iterrows():
                    # 必須項目「Lot番号」が空ならスキップ
                    if not row.get('Lot番号') or pd.isna(row.get('Lot番号')):
                        continue

                    # ★ 必須項目「実験日」が空ならスキップ
                    raw_date = row.get('実験日')
                    if not raw_date or pd.isna(raw_date):
                        continue
                        
                    # 実験日のパース（日付として認識できない文字が入っていた場合もスキップ）
                    try:
                        recorded_date = pd.to_datetime(raw_date).date()
                    except:
                        continue

                    # JSONに入れる動的データの抽出
                    experimental_data = {}
                    for item_name in item_names:
                        if item_name in df.columns:
                            val = row.get(item_name)
                            # 空欄（NaNやNone）じゃない場合だけ処理
                            if pd.notna(val) and val is not None:
                                # ★魔法の1行：Numpy型(int64, float64等)をPython標準型に変換！
                                if hasattr(val, 'item'):
                                    val = val.item()
                                
                                experimental_data[item_name] = val

                    # ＝＝＝ ↓ 新しい「上書き＆追記」の処理 ↓ ＝＝＝
                    existing_lot = LotData.objects.filter(theme=theme, lot_number=row.get('Lot番号')).first()

                    if existing_lot:
                        # 【上書き・追記モード】すでにデータが存在する場合
                        if raw_date:
                            existing_lot.recorded_date = recorded_date
                        if row.get('評価メモ') is not None:
                            existing_lot.evaluation_memo = row.get('評価メモ')
                            
                        # 動的データ（JSON）は合体（マージ）させる
                        current_data = existing_lot.experimental_data or {}
                        current_data.update(experimental_data)
                        existing_lot.experimental_data = current_data
                        
                        existing_lot.save() # 更新を保存
                    else:
                        # 【新規作成モード】データが存在しない場合
                        LotData.objects.create(
                            theme=theme,
                            lot_number=row.get('Lot番号'),
                            recorded_date=recorded_date,
                            evaluation_memo=row.get('評価メモ') or '',
                            experimental_data=experimental_data
                        )
                    
                    import_count += 1

                messages.success(request, f'{import_count}件のデータをインポート（または更新）しました！')
                return redirect('lot_list', pk=theme.pk)

            except Exception as e:
                messages.error(request, f'読み込みエラーが発生しました。ファイル形式を確認してください。（詳細: {e}）')
        
        return self.render_to_response({'form': form, 'theme': theme})


class ThemeItemImportView(UpdateView):
    template_name = 'experiment/theme_item_import.html'
    
    def get(self, request, pk):
        theme = get_object_or_404(ExperimentTheme, pk=pk)
        form = DataImportForm()
        return self.render_to_response({'form': form, 'theme': theme})

    def post(self, request, pk):
        theme = get_object_or_404(ExperimentTheme, pk=pk)
        form = DataImportForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['file']
            try:
                # pandas で1行目(ヘッダー)だけを読み込む (nrows=0)
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='shift_jis', nrows=0)
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='utf-8', nrows=0)
                elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file, nrows=0)
                else:
                    messages.error(request, 'CSVまたはExcelファイルを選択してください。')
                    return self.render_to_response({'form': form, 'theme': theme})

                # 除外する基本項目（これらは動的項目としては登録しない）
                exclude_cols = ['Lot番号', '実験日', '評価メモ', '担当者']
                
                # 現在すでに登録されている項目名を取得（重複登録を防ぐため）
                existing_items = ThemeItemDefinition.objects.filter(theme=theme).values_list('item_name', flat=True)
                
                # 現在の並び順の最後尾を取得
                next_order = ThemeItemDefinition.objects.filter(theme=theme).count()

                added_count = 0
                for col in df.columns:
                    col_name = str(col).strip()
                    # 無名列、基本項目、すでに存在する項目はスキップ
                    if not col_name or 'Unnamed' in col_name or col_name in exclude_cols or col_name in existing_items:
                        continue
                    
                    # 新しい項目としてデータベースに登録
                    ThemeItemDefinition.objects.create(
                        theme=theme,
                        order=next_order,
                        item_name=col_name,
                        data_type='text', # 一旦すべて「文字」として仮登録する
                        use_for_similarity=False,
                        use_for_anomaly=False,
                        is_active=True
                    )
                    next_order += 1
                    added_count += 1

                messages.success(request, f'{added_count}個の項目を追加しました！「データの種類」や「AI対象」を修正して保存してください。')
                # 抽出完了後は、データ項目設定画面にリダイレクトしてユーザーに修正させる
                return redirect('theme_settings', pk=theme.pk)

            except Exception as e:
                messages.error(request, f'読み込みエラーが発生しました。（詳細: {e}）')
        
        return self.render_to_response({'form': form, 'theme': theme})

# lab_system/experiment/views.py に追加

class LotUpdateView(UpdateView):
    model = LotData
    form_class = LotDataForm
    template_name = 'experiment/lot_form.html' # CreateViewと同じテンプレートを使い回せます

    def get_theme(self):
        return get_object_or_404(ExperimentTheme, pk=self.kwargs.get('theme_pk'))

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['theme'] = self.get_theme()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['theme'] = self.get_theme()
        return context

    def form_valid(self, form):
        # JSONデータの保存処理（CreateViewと同じ）
        experimental_data = {}
        for field_name, value in form.cleaned_data.items():
            if field_name.startswith('dynamic_'):
                real_item_name = field_name.replace('dynamic_', '')
                experimental_data[real_item_name] = value
        form.instance.experimental_data = experimental_data
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('lot_list', kwargs={'pk': self.get_theme().pk})

class ThemeExportView(View):
    """ テーマ一覧をCSVダウンロードするビュー """
    def get(self, request, *args, **kwargs):
        # 現在の日付 (YYYYMMDD形式) を取得
        today = datetime.now().strftime('%Y%m%d')
        
        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        
        # ファイル名の先頭に日付を付与
        filename = urllib.parse.quote(f"{today}_テーマ一覧.csv")
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
        
        writer = csv.writer(response)
        
        # ヘッダーに「作成者」と「更新日時」を追加
        writer.writerow(['テーマID', 'テーマ名', '作成者', '作成日時', '更新日時'])
        
        from .models import ExperimentTheme
        themes = ExperimentTheme.objects.all().order_by('-updated_at')
        
        for theme in themes:
            writer.writerow([
                theme.pk, 
                theme.name, 
                theme.author if theme.author else '', # 作成者を追加
                theme.created_at.strftime('%Y/%m/%d %H:%M') if theme.created_at else '',
                theme.updated_at.strftime('%Y/%m/%d %H:%M') if theme.updated_at else '' # 更新日時を追加
            ])
            
        return response


class LotDataExportView(View):
    """ 特定のテーマのLotデータをCSVダウンロードするビュー """
    def get(self, request, pk, *args, **kwargs):
        from .models import ExperimentTheme, LotData, ThemeItemDefinition
        
        theme = ExperimentTheme.objects.get(pk=pk)
        
        # 現在の日付 (YYYYMMDD形式) を取得
        today = datetime.now().strftime('%Y%m%d')
        
        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        
        # ファイル名の先頭に日付を付与 (例: 20260406_テーマ名_データ.csv)
        filename = urllib.parse.quote(f"{today}_{theme.name}_データ.csv")
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
        
        writer = csv.writer(response)
        
        item_defs = ThemeItemDefinition.objects.filter(theme=theme, is_active=True).order_by('order')
        headers = ['Lot番号', '実験日'] + [item.item_name for item in item_defs] + ['担当者', '評価メモ', '登録日時']
        writer.writerow(headers)
        
        lots = LotData.objects.filter(theme=theme).order_by('-recorded_date')
        
        for lot in lots:
            row = [
                lot.lot_number,
                lot.recorded_date.strftime('%Y/%m/%d') if lot.recorded_date else '',
            ]
            
            for item in item_defs:
                if lot.experimental_data and item.item_name in lot.experimental_data:
                    row.append(lot.experimental_data[item.item_name])
                else:
                    row.append('')
                    
            row.append(lot.recorded_by if lot.recorded_by else '')
            row.append(lot.evaluation_memo if lot.evaluation_memo else '')
            row.append(lot.created_at.strftime('%Y/%m/%d %H:%M') if lot.created_at else '')
            
            writer.writerow(row)
            
        return response



class LotAnalysisView(TemplateView):
    """ 詳細分析ダッシュボード（全データ転送・フロントエンド計算型） """
    template_name = 'experiment/lot_analysis.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        from .models import ExperimentTheme, ThemeItemDefinition, LotData
        
        theme = get_object_or_404(ExperimentTheme, pk=self.kwargs.get('pk'))
        context['theme'] = theme
        
        item_defs = ThemeItemDefinition.objects.filter(
            theme=theme, is_active=True
        ).order_by('order')
        context['item_defs'] = item_defs

        lots = LotData.objects.filter(theme=theme).order_by('recorded_date')
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