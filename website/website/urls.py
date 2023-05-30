from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from home import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('home.urls')),
    ]

# if settings.DEBUG:
#     import debug_toolbar
#     urlpatterns += [path('__debug', include(debug_toolbar.urls))]
