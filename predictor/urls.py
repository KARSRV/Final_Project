from django.urls import path
from .views import home, symptom_check, rag_chat, about, publication

urlpatterns = [
    path("", home, name="home"),
    path("symp/", symptom_check, name="symptom_check"),
    path("rag/", rag_chat, name="rag_chat"),   
    path("about/", about, name="about"),
    path("publication/", publication, name="publication"),       
]
