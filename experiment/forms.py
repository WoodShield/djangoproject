from django import forms
from .models import LotData, DatabaseItemDefinition, Database, UserProfile, Organization,Workspace
from django.forms.widgets import ClearableFileInput
from django.db.models import Q
from django.contrib.auth.models import User



class MultipleFileInput(ClearableFileInput):
    allow_multiple_selected = True
    
    # ブラウザから来た複数のファイルを正しくリストとして受け取る処理
    def value_from_datadict(self, data, files, name):
        if hasattr(files, 'getlist'):
            return files.getlist(name)
        return super().value_from_datadict(data, files, name)

# 複数の画像をチェックして受け入れる専用フィールド
class MultipleImageField(forms.ImageField):
    widget = MultipleFileInput()

    def clean(self, data, initial=None):
        if isinstance(data, (list, tuple)):
            return [super(MultipleImageField, self).clean(d, initial) for d in data]
        return super().clean(data, initial)

class LotDataForm(forms.ModelForm):
    class Meta:
        model = LotData

        fields = ['lot_number', 'recorded_date', 'recorded_by', 'evaluation_memo', 'is_anomaly_acknowledged']
        
        widgets = {
            'recorded_date': forms.DateInput(attrs={'type': 'date'}),
            'evaluation_memo': forms.Textarea(attrs={'rows': 3, 'placeholder': '実験の所見や特記事項があれば入力してください'}),
        }
    
    def clean_lot_number(self):
        lot_number = self.cleaned_data.get('lot_number')
        
        # 新規作成時など、databaseがまだ紐づいていない場合はスキップ
        if not self.database:
            return lot_number

        # 同じテーマ内に、同じLot番号がないか探す
        qs = LotData.objects.filter(database=self.database, lot_number=lot_number)
        
        # 編集（更新）の時は、「自分自身のLotデータ」は重複チェックから除外する
        if self.instance and self.instance.pk:
            qs = qs.exclude(pk=self.instance.pk)

        # もし既に存在していたら、赤文字でエラーを返す
        if qs.exists():
            raise forms.ValidationError(f"Lot番号「{lot_number}」は既に登録されています。再試験などの場合は、末尾に枝番（-2、-再 など）を付けてください。")

        return lot_number

    def __init__(self, *args, **kwargs):
        # views.pyから渡される 'database' を受け取る
        self.database = kwargs.pop('database', None)
        super().__init__(*args, **kwargs)

        if self.database:
            # このテーマに紐づく有効な項目定義を、並び順通りに取得
            items = DatabaseItemDefinition.objects.filter(database=self.database, is_active=True).order_by('order')
            
            for item in items:
                # フォームの内部的な名前を "dynamic_1", "dynamic_2" のように設定
                field_name = f"dynamic_{item.item_name}"
                
                # データの種類に合わせてフォームの部品（Widget）を変える魔法
                if item.data_type == 'text':
                    self.fields[field_name] = forms.CharField(label=item.item_name, required=False)
                
                elif item.data_type == 'number':
                    self.fields[field_name] = forms.FloatField(label=item.item_name, required=False)
                
                elif item.data_type == 'calc':
                    # 計算フィールドはユーザーに入力させないため readonly にする
                    self.fields[field_name] = forms.FloatField(
                        label=f"{item.item_name} 🤖(自動計算)", 
                        required=False,
                        widget=forms.NumberInput(attrs={'readonly': 'readonly', 'class': 'bg-light'})
                    )
                
                elif item.data_type == 'image':
                    self.fields[field_name] = MultipleImageField(
                        label=item.item_name, required=False
                    )
                
                elif item.data_type == 'date':
                    self.fields[field_name] = forms.DateField(
                        label=item.item_name, required=False, widget=forms.DateInput(attrs={'type': 'date'})
                    )
                
                elif item.data_type == 'select':
                    choices = [('', '---------')]
                    if item.choices_text:
                        # カンマ区切りの文字列をリストにして選択肢を作る
                        choices += [(c.strip(), c.strip()) for c in item.choices_text.split(',')]
                    self.fields[field_name] = forms.ChoiceField(label=item.item_name, choices=choices, required=False)
                
                elif item.data_type == 'checkbox':
                    self.fields[field_name] = forms.BooleanField(label=item.item_name, required=False)
                
                elif item.data_type == 'long_text':
                    self.fields[field_name] = forms.CharField(
                        label=item.item_name, required=False, widget=forms.Textarea(attrs={'rows': 2})
                    )
                
                if item.calculation_formula:
                    # widgetの属性(attrs)に 'data-formula' を追加し、JSが検知できるようにする
                    self.fields[field_name].widget.attrs['data-formula'] = item.calculation_formula

        if self.instance and self.instance.pk and self.instance.experimental_data:
            # JSON (experimental_data) の中身を1つずつ取り出す
            for key, value in self.instance.experimental_data.items():
                field_name = f'dynamic_{key}'
                
                if field_name in self.fields:
                    # ★修正：新しく作った MultipleImageField かどうかで処理を分ける
                    if self.fields[field_name].__class__.__name__ == 'MultipleImageField':
                        # 画像リストの場合は、HTML側で「現在の画像」として表示するためにセット
                        self.fields[field_name].existing_images = value
                    else:
                        # それ以外（テキストや数値など）は、入力欄に初期値をセット
                        self.initial[field_name] = value
        
        # 1. 類似性検索の残骸は先ほど削除したため、anomaly関連のみ残しています
        if not self.instance or not self.instance.pk or not self.instance.ai_anomaly_msg:
            self.fields.pop('is_anomaly_acknowledged', None)

        # 2. 警告が出ている時は、チェックボックスの下に「AIからの警告文」を表示してあげる
        if self.instance and self.instance.pk:
            if self.instance.ai_anomaly_msg and 'is_anomaly_acknowledged' in self.fields:
                self.fields['is_anomaly_acknowledged'].help_text = f"<div class='text-danger fw-bold mt-1'>{self.instance.ai_anomaly_msg}</div>"

class ThemeForm(forms.ModelForm):
    class Meta:
        model = Database
        # 類似検索は廃止したため、enable_similarity_search は外す
        fields = ['name', 'author', 'enable_anomaly_check']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # チェックがOFFでもエラーで弾かれないようにする
        if 'enable_anomaly_check' in self.fields:
            self.fields['enable_anomaly_check'].required = False



class DataImportForm(forms.Form):
    file = forms.FileField(
        label='CSVまたはExcelファイル',
        help_text='※1行目は項目名（Lot番号, 実験日, 評価メモ, 各データ項目名）にしてください。'
    )



class DatabaseItemDefinitionForm(forms.ModelForm):
    class Meta:
        model = DatabaseItemDefinition
        fields = ['order', 'item_name', 'data_type', 'choices_text', 'calculation_formula',  'use_for_anomaly', 'is_active']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
            # チェックがOFFでもエラーで弾かれないようにする
        if 'use_for_anomaly' in self.fields:
            self.fields['use_for_anomaly'].required = False
        if 'is_active' in self.fields:
            self.fields['is_active'].required = False

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['display_name', 'organization']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['display_name'].widget.attrs.update({
            'class': 'form-control', 
            'placeholder': '画面に表示される名前を入力'
        })
        self.fields['organization'].widget.attrs.update({
            'class': 'form-select'
        })
        self.fields['organization'].empty_label = "未所属（選択してください）"

class WorkspaceForm(forms.ModelForm):
    class Meta:
        model = Workspace
        # 既存のフィールド（'name'等）の配列に、以下の2つを追加してください
        fields = ['name', 'allowed_orgs', 'allowed_users'] 
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        
        self.fields['allowed_orgs'].widget = forms.SelectMultiple(attrs={'class': 'form-select', 'size': '6'})
        self.fields['allowed_orgs'].queryset = Organization.objects.all()
        self.fields['allowed_orgs'].help_text = "指定しない場合は、全ユーザーに公開されます。"
        
        self.fields['allowed_users'].widget = forms.SelectMultiple(attrs={'class': 'form-select', 'size': '6'})
        self.fields['allowed_users'].queryset = User.objects.all()
        self.fields['allowed_users'].help_text = "部署に関わらず、個人単位で特別に許可したいユーザーを選択してください。"

class DatabaseForm(forms.ModelForm):
    class Meta:
        model = Database
        fields = ['name', 'author', 'enable_anomaly_check', 'allowed_orgs', 'allowed_users'] 
        
    def __init__(self, *args, **kwargs):
        # Viewから渡される 'workspace' の情報を受け取る
        self.workspace = kwargs.pop('workspace', None)
        super().__init__(*args, **kwargs)
        
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        if 'enable_anomaly_check' in self.fields:
            self.fields['enable_anomaly_check'].required = False
        
        self.fields['allowed_orgs'].widget = forms.SelectMultiple(attrs={'class': 'form-select', 'size': '6'})
        self.fields['allowed_users'].widget = forms.SelectMultiple(attrs={'class': 'form-select', 'size': '6'})
        self.fields['allowed_orgs'].help_text = "指定しない場合は、親ワークスペースの権限設定を引き継ぎます。"
        self.fields['allowed_users'].help_text = "部署に関わらず、個人単位で特別に許可したいユーザーを選択してください。"

        # =========================================================
        # ★ ワークスペースの権限に応じて選択肢（リストボックスの中身）を絞り込む
        # =========================================================
        if self.workspace:
            # ワークスペースが「全公開（制限なし）」の場合は、全員を選択可能にする
            if not self.workspace.allowed_orgs.exists() and not self.workspace.allowed_users.exists():
                self.fields['allowed_orgs'].queryset = Organization.objects.all()
                self.fields['allowed_users'].queryset = User.objects.all()
            else:
                # 1. 部署の選択肢：ワークスペースで許可されている部署のみに絞る
                self.fields['allowed_orgs'].queryset = self.workspace.allowed_orgs.all()
                
                # 2. ユーザーの選択肢：ワークスペースで個別許可されている人 ＋ 許可部署に所属している人 に絞る
                q_user = Q(id__in=self.workspace.allowed_users.all())
                q_org = Q(profile__organization__in=self.workspace.allowed_orgs.all())
                self.fields['allowed_users'].queryset = User.objects.filter(q_user | q_org).distinct()
        else:
            # 万が一ワークスペース情報がない場合は、安全のため選択肢を空にする
            self.fields['allowed_orgs'].queryset = Organization.objects.none()
            self.fields['allowed_users'].queryset = User.objects.none()