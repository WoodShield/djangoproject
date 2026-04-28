from django.views.generic import UpdateView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from ..models import UserProfile, Organization
from ..forms import UserProfileForm

class MyPageView(LoginRequiredMixin, UpdateView):
    """ユーザー自身のプロフィールと所属組織を設定するビュー"""
    model = UserProfile
    form_class = UserProfileForm
    template_name = 'experiment/mypage.html'
    success_url = reverse_lazy('workspace_list')

    def get_object(self, queryset=None):
        # ログイン中のユーザーのプロフィールを取得。まだ無ければ裏側で自動作成する。
        profile, created = UserProfile.objects.get_or_create(user=self.request.user)
        return profile

    def form_valid(self, form):
        messages.success(self.request, 'プロフィールを更新しました。')
        return super().form_valid(form)

class OrganizationCreateView(LoginRequiredMixin, CreateView):
    """新しい部署（組織）をマスターに登録するビュー"""
    model = Organization
    fields = ['name']
    template_name = 'experiment/organization_form.html'
    success_url = reverse_lazy('mypage') # 登録後はマイページに戻る

    def form_valid(self, form):
        # 登録成功時のメッセージ
        messages.success(self.request, f'新しい部署「{form.instance.name}」を登録しました。上のリストから選択して保存してください。')
        return super().form_valid(form)


class SignUpView(CreateView):
    """新規ユーザー登録ビュー"""
    form_class = UserCreationForm
    template_name = 'experiment/signup.html'
    # 登録後はマイページに飛ばして、名前と部署を設定させる
    success_url = reverse_lazy('mypage') 

    def form_valid(self, form):
        # ユーザーをデータベースに保存
        user = form.save()
        # 保存したユーザーでそのまま自動ログイン（再度ログイン画面に入力させない親切設計）
        login(self.request, user)
        messages.success(self.request, 'アカウントを作成しました！続けて「表示名」と「所属部署」を設定してください。')
        return super().form_valid(form)