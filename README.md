Face Recognition App
==========================================

Dieses Projekt implementiert ein Gesichtserkennungs- und Robotersteuerungssystem.

Installation
============

Es gibt zwei Methoden, um die benötigten Python-Pakete zu installieren:

1. Verwendung von `pip` mit einer `requirements.txt`-Datei
2. Verwendung von Conda mit einer `environment.yml`-Datei

Methode 1: Verwendung von `pip`
-------------------------------

1. Stellen Sie sicher, dass Python 3.9 und `pip` auf Ihrem System installiert sind.

2. Öffnen Sie ein Terminal und navigieren Sie zu Ihrem Projektverzeichnis.

3. Führen Sie den folgenden Befehl aus:
   pip install -r requirements.txt

Hinweis: Nachdem Sie deepface mit pip installiert haben, wird TensorFlow automatisch von deepface aktualisiert. Diese aktualisierte Version funktioniert jedoch nicht korrekt. Installieren Sie daher anschließend TensorFlow in der Version 2.9.1 mit folgendem Befehl:

pip install tensorflow==2.9.1

Sie müssen außerdem Mediapipe installieren. Beachten Sie jedoch, dass Mediapipe die Bibliotheken JAX und JAXlib mitinstalliert. Diese beiden Bibliotheken müssen Sie wieder deinstallieren, da der Code sonst nicht funktioniert:

pip uninstall jax jaxlib

Für YOLOv8 müssen Sie dann Ultralitytics installieren:

pip install ultralytics



Methode 2: Verwendung von Conda
-------------------------------
1- Stellen Sie sicher, dass Conda auf Ihrem System installiert ist.

2- Öffnen Sie ein Terminal und navigieren Sie zu Ihrem Projektverzeichnis.

3- Führen Sie die folgenden Befehle aus:

	conda env create -f environment.yml
	conda activate niryo_ned_env


Projekt Starten
================
Nach der Installation der Pakete können Sie das Projekt starten, indem Sie das Hauptskript ausführen:

python robot_gui.py










