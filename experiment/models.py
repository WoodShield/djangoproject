from django.db import models
from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
User = get_user_model()

# ==========================================
# 1. 組織とユーザー (新設)
# ==========================================
class Organization(models.Model):
    """ 物理的な会社組織・部署（例：開発1部 第2グループ） """
    name = models.CharField("部署・組織名", max_length=100, unique=True)
    created_at = models.DateTimeField("作成日", auto_now_add=True)

    class Meta:
        verbose_name = "組織・部署"
        verbose_name_plural = "組織・部署"

    def __str__(self):
        return self.name

class UserProfile(models.Model):
    """ ユーザー拡張情報（マイページ用） """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile', verbose_name="ユーザー")
    display_name = models.CharField("表示名", max_length=100)
    organization = models.ForeignKey(Organization, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="所属組織")

    class Meta:
        verbose_name = "ユーザープロフィール"
        verbose_name_plural = "ユーザープロフィール"

    def __str__(self):
        return self.display_name or self.user.username

# ==========================================
# 2. ワークスペース 
# ==========================================
class Workspace(models.Model):
    """ プロジェクトごとの論理的な共有スペース """
    name = models.CharField("ワークスペース名", max_length=100)
    description = models.TextField("説明", blank=True)
    creator = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="作成者")

    # 新しい権限管理フィールド
    allowed_orgs = models.ManyToManyField(
        'Organization', 
        blank=True, 
        related_name='allowed_workspaces', 
        verbose_name='許可する部署'
    )
    allowed_users = models.ManyToManyField(
        User, 
        blank=True, 
        related_name='allowed_workspaces', 
        verbose_name='許可するユーザー'
    )
    
    created_at = models.DateTimeField("作成日", auto_now_add=True)

    class Meta:
        verbose_name = "ワークスペース"
        verbose_name_plural = "ワークスペース"

    def can_access(self, user):
        """
        ユーザーがこのワークスペースにアクセスできるか判定する
        """
        # 管理者は常にフルアクセス
        if user.is_superuser:
            return True

        # 1. どちらも空（指定なし）なら全員アクセス可能（制限なし）
        if not self.allowed_orgs.exists() and not self.allowed_users.exists():
            return True
            
        # 2. 【重要】ユーザー個人が許可されていれば無条件でアクセス可能
        if self.allowed_users.filter(id=user.id).exists():
            return True
            
        # 3. ユーザーの所属部署が許可されていればアクセス可能
        if hasattr(user, 'profile') and user.profile.organization:
            if self.allowed_orgs.filter(id=user.profile.organization.id).exists():
                return True
                
        return False

    def __str__(self):
        return self.name

# ==========================================
# 3. データベース 
# ==========================================
class Database(models.Model):
    """ 実験データなどのテーブル本体 """
    workspace = models.ForeignKey(
        Workspace, 
        on_delete=models.CASCADE, 
        related_name='databases', 
        verbose_name="所属ワークスペース",
        null=True,
        blank=True
    )
    name = models.CharField("データベース名", max_length=200)
    author = models.CharField("作成者", max_length=100, blank=True, null=True)

    
    allowed_orgs = models.ManyToManyField(
        'Organization', 
        blank=True, 
        related_name='allowed_databases', 
        verbose_name='許可する部署'
    )
    allowed_users = models.ManyToManyField(
        User, 
        blank=True, 
        related_name='allowed_databases', 
        verbose_name='許可するユーザー'
    )

    def can_access(self, user):
        """
        ユーザーがこのデータベースにアクセスできるか判定する
        """
        # 0. 大前提：親であるWorkspace自体にアクセスできなければ、中身も見せない
        if not self.workspace.can_access(user):
            return False

        # 1. データベース自体の権限指定がどちらも空なら、Workspaceの権限を引き継ぐ（アクセス可能）
        if not self.allowed_orgs.exists() and not self.allowed_users.exists():
            return True
            
        # 2. 【重要】ユーザー個人が許可されていればアクセス可能（部署NGでもユーザーOKならTrue）
        if self.allowed_users.filter(id=user.id).exists():
            return True
            
        # 3. ユーザーの所属部署が許可されていればアクセス可能
        if hasattr(user, 'profile') and user.profile.organization:
            if self.allowed_orgs.filter(id=user.profile.organization.id).exists():
                return True
                
        # 上記のどれにも当てはまらない場合はアクセス不可
        return False
    
    
    # --- AI・機械学習（scikit-learn）機能のON/OFFスイッチ ---
    enable_anomaly_check = models.BooleanField(
        "入力異常検知を有効にする（品管・定型業務向け）", 
        default=False,
        help_text="Isolation Forestによる桁間違いや外れ値の警告を出す場合はチェックを入れてください。"
    )
    
    created_at = models.DateTimeField("作成日", auto_now_add=True)
    updated_at = models.DateTimeField("更新日", auto_now=True)

    class Meta:
        verbose_name = "データベース"
        verbose_name_plural = "データベース"

    def __str__(self):
        return self.name

# ==========================================
# 4. データ項目定義 
# ==========================================
class DatabaseItemDefinition(models.Model):
    """ データベースごとの入力項目（例：温度、外観、成分Aなど）を定義するモデル """
    database = models.ForeignKey(Database, on_delete=models.CASCADE, related_name='items')
    item_name = models.CharField("項目名", max_length=100)
    
    DATA_TYPE_CHOICES = (
        ('text', '文字（短文）'),
        ('number', '数字'),
        ('calc', '計算（自動算出）'),
        ('image', '画像'),
        ('file', 'ファイル添付'),
        ('select', '選択'),
        ('checkbox', 'チェックボックス（ON/OFF）'),
        ('date', '日付'),
        ('time', '時刻'),
        ('long_text', '長文テキスト（所感など）'),
    )
    data_type = models.CharField("データの種類", max_length=20, choices=DATA_TYPE_CHOICES)
    
    choices_text = models.CharField("選択肢", max_length=255, blank=True, null=True, help_text="カンマ区切りで入力")
    
    calculation_formula = models.CharField(
        "計算式", 
        max_length=255, 
        blank=True, 
        null=True, 
        help_text="例: {処理後の重量} - {風袋重量}"
    )
    
    use_for_anomaly = models.BooleanField("異常検知(AI)の対象", default=True)
    is_active = models.BooleanField("有効フラグ", default=True)
    order = models.IntegerField("並び順", default=0)

    class Meta:
        ordering = ['order', 'id']
        verbose_name = "データ項目定義"
        verbose_name_plural = "データ項目定義"

    def __str__(self):
        return f"{self.database.name} - {self.item_name} ({self.get_data_type_display()})"

# ==========================================
# 5. Lotデータ
# ==========================================
class LotData(models.Model):
    """ 各データのLot別データ（日々の実験結果） """
    database = models.ForeignKey(Database, on_delete=models.CASCADE, verbose_name="データベース", related_name="lots")
    lot_number = models.CharField("Lot番号", max_length=50)
    recorded_date = models.DateField("実験日")
    recorded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, verbose_name="担当者")
    
    experimental_data = models.JSONField("実験データ（条件・結果）", default=dict, blank=True)
    
    EVALUATION_CHOICES = (
        ('A', 'A: 良好 / 成功'),
        ('B', 'B: やや不良 / 改善の余地あり'),
        ('C', 'C: 不良 / 失敗'),
    )
    evaluation_score = models.CharField("評定", max_length=10, choices=EVALUATION_CHOICES, null=True, blank=True)
    evaluation_memo = models.TextField("評価メモ（所見など）", blank=True)
    
    created_at = models.DateTimeField("作成日", auto_now_add=True)
    updated_at = models.DateTimeField("更新日", auto_now=True)
    ai_anomaly_msg = models.TextField(blank=True, null=True, verbose_name="異常検知メッセージ")
    is_anomaly_acknowledged = models.BooleanField(default=False, verbose_name="この異常警告を「問題なし」として非表示にする")

    class Meta:
        verbose_name = "Lotデータ"
        verbose_name_plural = "Lotデータ"
        constraints = [
            models.UniqueConstraint(
                fields=['database', 'lot_number'], 
                name='unique_lot_per_database'
            )
        ]

    def __str__(self):
        return f"{self.database.name} - {self.lot_number}"

# ==========================================
# 6. 学習済みモデル
# ==========================================
class TrainedModelMetadata(models.Model):
    """ 学習済みモデルのメタデータとファイルパスを管理するモデル """
    database = models.ForeignKey(
        Database, 
        on_delete=models.CASCADE, 
        related_name='trained_models', 
        verbose_name="対象データベース"
    )
    model_file = models.FileField("モデルファイル", upload_to='mi_models/')
    
    target_variable = models.CharField("目的変数(Y)", max_length=100)
    features_list = models.JSONField("説明変数(X)リスト", default=list)
    
    preprocessing_meta = models.JSONField(
        "前処理メタデータ", 
        default=dict, 
        blank=True,
        help_text="欠損値処理、データ型判定、カテゴリエンコーディングの相関情報など"
    )
    
    metrics = models.JSONField(
        "評価メトリクス", 
        default=dict, 
        blank=True,
        help_text="R2, RMSEなどのCVスコア"
    )
    
    created_at = models.DateTimeField("作成日", auto_now_add=True)

    class Meta:
        verbose_name = "学習済みモデルメタデータ"
        verbose_name_plural = "学習済みモデルメタデータ"
        ordering = ['-created_at']

    def __str__(self):
        r2_score = self.metrics.get('r2', 'N/A')
        if isinstance(r2_score, float):
            r2_score = round(r2_score, 3)
        return f"{self.database.name} - Y:{self.target_variable} (R2: {r2_score})"