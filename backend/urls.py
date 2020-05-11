from django.contrib import admin
from django.urls import include, path
from lidar.views import lidarApi

urlpatterns = [
    path('lidar/', lidarApi.as_view()),
    path('admin/', admin.site.urls),
]
