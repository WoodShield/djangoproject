from django.contrib import admin
from django.urls import path, include
# ＝＝＝ ★追加1：settings と static を読み込む ＝＝＝
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    # experimentアプリのurls.pyを読み込む
    path('', include('experiment.urls')),
]

# ＝＝＝ ★追加2：開発環境(DEBUG=True)のときだけ、画像のURLを有効にする ＝＝＝
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)