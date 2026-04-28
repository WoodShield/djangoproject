from django.views.generic import UpdateView
from django.views import View
from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from django.http import HttpResponse
from datetime import datetime
import pandas as pd
import csv
import urllib.parse

from ..models import Database, LotData, DatabaseItemDefinition
from ..forms import DataImportForm

class LotDataImportView(UpdateView):
    template_name = 'experiment/lot_import.html'
    
    def get(self, request, pk):
        database = get_object_or_404(Database, pk=pk)
        form = DataImportForm()
        return self.render_to_response({'form': form, 'database': database})

    def post(self, request, pk):
        database = get_object_or_404(Database, pk=pk)
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
                    return self.render_to_response({'form': form, 'database': database})

                # NaN（空のセル）を None に変換
                df = df.where(pd.notnull(df), None)

                # 設定画面で作った項目の一覧を取得
                item_defs = DatabaseItemDefinition.objects.filter(database=database, is_active=True)
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
                    existing_lot = LotData.objects.filter(database=database, lot_number=row.get('Lot番号')).first()

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
                            database=database,
                            lot_number=row.get('Lot番号'),
                            recorded_date=recorded_date,
                            evaluation_memo=row.get('評価メモ') or '',
                            experimental_data=experimental_data
                        )
                    
                    import_count += 1

                messages.success(request, f'{import_count}件のデータをインポート（または更新）しました！')
                return redirect('lot_list', pk=database.pk)

            except Exception as e:
                messages.error(request, f'読み込みエラーが発生しました。ファイル形式を確認してください。（詳細: {e}）')
        
        return self.render_to_response({'form': form, 'database': database})

class ThemeItemImportView(UpdateView):
    template_name = 'experiment/database_item_import.html'
    
    def get(self, request, pk):
        database = get_object_or_404(Database, pk=pk)
        form = DataImportForm()
        return self.render_to_response({'form': form, 'database': database})

    def post(self, request, pk):
        database = get_object_or_404(Database, pk=pk)
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
                    return self.render_to_response({'form': form, 'database': database})

                # 除外する基本項目（これらは動的項目としては登録しない）
                exclude_cols = ['Lot番号', '実験日', '評価メモ', '担当者']
                
                # 現在すでに登録されている項目名を取得（重複登録を防ぐため）
                existing_items = DatabaseItemDefinition.objects.filter(database=database).values_list('item_name', flat=True)
                
                # 現在の並び順の最後尾を取得
                next_order = DatabaseItemDefinition.objects.filter(database=database).count()

                added_count = 0
                for col in df.columns:
                    col_name = str(col).strip()
                    # 無名列、基本項目、すでに存在する項目はスキップ
                    if not col_name or 'Unnamed' in col_name or col_name in exclude_cols or col_name in existing_items:
                        continue
                    
                    # 新しい項目としてデータベースに登録
                    DatabaseItemDefinition.objects.create(
                        database=database,
                        order=next_order,
                        item_name=col_name,
                        data_type='text', # 一旦すべて「文字」として仮登録する
                        
                        use_for_anomaly=False,
                        is_active=True
                    )
                    next_order += 1
                    added_count += 1

                messages.success(request, f'{added_count}個の項目を追加しました！「データの種類」や「AI対象」を修正して保存してください。')
                # 抽出完了後は、データ項目設定画面にリダイレクトしてユーザーに修正させる
                return redirect('database_settings', pk=database.pk)

            except Exception as e:
                messages.error(request, f'読み込みエラーが発生しました。（詳細: {e}）')
        
        return self.render_to_response({'form': form, 'database': database})

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
        
        databases = Database.objects.all().order_by('-updated_at')
        
        for database in databases:
            writer.writerow([
                database.pk, 
                database.name, 
                database.author if database.author else '', # 作成者を追加
                database.created_at.strftime('%Y/%m/%d %H:%M') if database.created_at else '',
                database.updated_at.strftime('%Y/%m/%d %H:%M') if database.updated_at else '' # 更新日時を追加
            ])
            
        return response

class LotDataExportView(View):
    """ 特定のテーマのLotデータをCSVダウンロードするビュー """
    def get(self, request, pk, *args, **kwargs):

        
        database = Database.objects.get(pk=pk)
        
        # 現在の日付 (YYYYMMDD形式) を取得
        today = datetime.now().strftime('%Y%m%d')
        
        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        
        # ファイル名の先頭に日付を付与 (例: 20260406_テーマ名_データ.csv)
        filename = urllib.parse.quote(f"{today}_{database.name}_データ.csv")
        response['Content-Disposition'] = f"attachment; filename*=UTF-8''{filename}"
        
        writer = csv.writer(response)
        
        item_defs = DatabaseItemDefinition.objects.filter(database=database, is_active=True).order_by('order')
        headers = ['Lot番号', '実験日'] + [item.item_name for item in item_defs] + ['担当者', '評価メモ', '登録日時']
        writer.writerow(headers)
        
        lots = LotData.objects.filter(database=database).order_by('-recorded_date')
        
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
