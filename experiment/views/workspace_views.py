from django.views.generic import ListView, CreateView, UpdateView
from django.urls import reverse_lazy
from ..models import Workspace
from django.db.models import Q
from ..forms import WorkspaceForm
from django.contrib.auth.mixins import LoginRequiredMixin



class WorkspaceListView(LoginRequiredMixin, ListView):
    model = Workspace
    template_name = 'experiment/workspace_list.html'
    context_object_name = 'workspaces'

    def get_queryset(self):
        user = self.request.user
        if user.is_superuser:
            return Workspace.objects.all().order_by('-created_at')

        # ユーザーの所属組織を取得
        org = user.profile.organization if hasattr(user, 'profile') else None

        # 検索条件の組み立て（OR条件）
        q_free = Q(allowed_orgs__isnull=True, allowed_users__isnull=True) # 制限なし
        q_user = Q(allowed_users=user) # ユーザー個人が許可
        q_org = Q(allowed_orgs=org) if org else Q(pk__in=[]) # 部署が許可

        # 権限があるものだけを抽出して返す
        return Workspace.objects.filter(q_free | q_user | q_org).distinct().order_by('-created_at')

class WorkspaceCreateView(LoginRequiredMixin, CreateView):
    model = Workspace
    form_class = WorkspaceForm # ← fields ではなくフォームクラスを使う
    template_name = 'experiment/workspace_form.html'
    success_url = reverse_lazy('workspace_list')

    def form_valid(self, form):
        form.instance.creator = self.request.user
        return super().form_valid(form)

class WorkspaceUpdateView(LoginRequiredMixin, UpdateView):
    model = Workspace
    form_class = WorkspaceForm
    template_name = 'experiment/workspace_form.html' # 作成画面と共通でOK
    success_url = reverse_lazy('workspace_list')