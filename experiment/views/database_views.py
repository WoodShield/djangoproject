from django.views.generic import ListView, CreateView, UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse
from django.shortcuts import get_object_or_404, redirect
from django.forms.models import inlineformset_factory, BaseInlineFormSet
from django.contrib import messages
from django.core.exceptions import ValidationError, PermissionDenied
from django.db.models import Q



# 1つ上の階層(..)からモデルとフォームを読み込む
from ..models import Database, DatabaseItemDefinition, Workspace
from ..forms import DatabaseItemDefinitionForm, DatabaseForm # 新しいDatabaseFormを使用

class ThemeListView(LoginRequiredMixin, ListView):
    """データベース一覧表示（権限フィルター付き）"""
    model = Database
    template_name = 'experiment/database_list.html'
    context_object_name = 'databases'

    def get_queryset(self):
        dept_pk = self.kwargs.get('dept_pk')
        workspace = get_object_or_404(Workspace, pk=dept_pk)
        user = self.request.user

        # 【防壁】親のWorkspaceに権限がない場合は403エラー
        if not workspace.can_access(user):
            raise PermissionDenied("このワークスペースへのアクセス権限がありません。")

        # ワークスペース内のデータベースに絞り込み
        qs = Database.objects.filter(workspace=workspace)

        if user.is_superuser:
            return qs.order_by('-updated_at')

        # データベースごとの権限フィルタリング
        org = user.profile.organization if hasattr(user, 'profile') else None
        q_free = Q(allowed_orgs__isnull=True, allowed_users__isnull=True)
        q_user = Q(allowed_users=user)
        q_org = Q(allowed_orgs=org) if org else Q(pk__in=[])

        return qs.filter(q_free | q_user | q_org).distinct().order_by('-updated_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['workspace'] = get_object_or_404(Workspace, pk=self.kwargs.get('dept_pk'))
        return context

class ThemeCreateView(LoginRequiredMixin, CreateView):
    """データベース新規作成"""
    model = Database
    form_class = DatabaseForm # 新しく作った権限設定付きのフォームを使用
    template_name = 'experiment/database_form.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # URLの 'dept_pk' からワークスペースを取得して渡す
        context['workspace'] = get_object_or_404(Workspace, pk=self.kwargs.get('dept_pk'))
        return context

    # フォームにワークスペースの情報を渡す  
    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['workspace'] = get_object_or_404(Workspace, pk=self.kwargs.get('dept_pk'))
        return kwargs

    def form_valid(self, form):
        workspace_id = self.kwargs.get('dept_pk')
        workspace = get_object_or_404(Workspace, pk=workspace_id)
        
        # 親ワークスペースに権限がないユーザーが作成しようとしたらブロック
        if not workspace.can_access(self.request.user):
            raise PermissionDenied("権限がありません。")

        form.instance.workspace = workspace
        form.instance.author = self.request.user.username
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('database_list', kwargs={'dept_pk': self.kwargs.get('dept_pk')})

# ==========================================
# 項目設定（FormSet）関連のロジック
# ==========================================
class BaseThemeItemFormSet(BaseInlineFormSet):
    def clean(self):
        super().clean()
        if any(self.errors):
            return 
        
        item_names = []
        for form in self.forms:
            if self.can_delete and self._should_delete_form(form):
                continue
            
            name = form.cleaned_data.get('item_name')
            if name:
                if name in item_names:
                    raise ValidationError(f"⚠️ エラー：項目名「{name}」が重複して入力されています。別の名前に変更してください。")
                item_names.append(name)

ItemFormSet = inlineformset_factory(
    Database, DatabaseItemDefinition,
    form=DatabaseItemDefinitionForm,
    formset=BaseThemeItemFormSet,
    extra=1, 
    can_delete=True
)


class DatabaseUpdateView(LoginRequiredMixin, UpdateView):
    """データベースの基本設定と権限の編集"""
    model = Database
    form_class = DatabaseForm
    template_name = 'experiment/database_form.html' # 新規作成と同じ画面（左右リスト付き）を使い回す

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['workspace'] = self.object.workspace
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['workspace'] = self.object.workspace
        return context

    def get_success_url(self):
        return reverse('lot_list', kwargs={'pk': self.object.pk})


class ThemeSettingsView(LoginRequiredMixin, UpdateView):
    """データベースの項目定義（列の追加・編集）専用のビュー"""
    model = Database
    fields = []  # ★重要：データベース本体のフィールドはここでは編集しない
    template_name = 'experiment/database_settings.html'

    # get_form_kwargs は DatabaseForm を使わないので削除してOKです

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
        
        # データベース本体(form)は何も変更しないが、
        # 保存処理の流れとして一応実行し、その後に項目(formset)を保存する
        if item_formset.is_valid():
            self.object = form.save()
            item_formset.instance = self.object
            item_formset.save()
            messages.success(self.request, 'データ項目の設定を保存しました。')
            
            action = self.request.POST.get('submit_action')
            if action == 'return':
                return redirect('lot_list', pk=self.object.pk)
            else:
                return redirect('database_settings', pk=self.object.pk)
        else:
            # エラー表示ロジックはそのまま継続
            error_details = []
            for i, form_errors in enumerate(item_formset.errors):
                if form_errors:
                    detail = ", ".join([f"{k}: {v}" for k, v in form_errors.items()])
                    error_details.append(f"【{i+1}行目】 {detail}")
            if item_formset.non_form_errors():
                error_details.append(f"【全体】 {item_formset.non_form_errors()}")
            
            error_msg = '入力内容にエラーがあります。詳細: ' + " / ".join(error_details)
            messages.error(self.request, error_msg)
            return self.render_to_response(self.get_context_data(form=form))