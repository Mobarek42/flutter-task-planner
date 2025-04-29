Projet PFE : Application de Planification de Tâches
Application de planification de tâches avec un frontend Flutter et un backend Flask, optimisée via PuLP et Simulated Annealing.
Structure

frontend/ : Code Flutter (Dart).
backend/ : Code Flask (Python).

Prérequis

Flutter SDK (3.22.0 recommandé).
Python 3.x.
GitHub Codespaces ou environnement local.

Exécution dans Codespaces

Backend (Flask) :
cd backend
pip install -r requirements.txt
python app.py

Note l'URL publique ( https://<votre-codespace>.github.dev:3000).

Frontend (Flutter) :
cd frontend
flutter pub get
flutter run -d web-server --web-port 8080 --release

Accède via l'URL publique ( https://<votre-codespace>.github.dev:8080).

Configuration :

Mets à jour _baseUrl dans frontend/lib/main.dart avec l'URL du backend.



Exécution locale

Clone le dépôt :git clone <URL-du-dépôt>


Backend :cd backend
pip install -r requirements.txt
python app.py


Frontend :cd frontend
flutter pub get
flutter run -d chrome


Configuration : Ajuste _baseUrl dans main.dart ('https://flutter-task-planner-3.onrender.com').


