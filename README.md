# Workflow für anwendung und erweiterung eines Object Detection Models

## Ordnerstruktur

```
.
├── INSTALLATION.md
├── README.md
└── object_detection
    ├── images
    │   └── example.png
    └── plot_object_detection_saved_model.ipynb
```

## Zusammenfassung

In diesem Projekt wird mittels Transferlearning ein etablierstes Object Detection Model verwendet und mit eigenen
Bildern
(von Kindern) erweitert. Dieses neue Model soll dazu dienen Objekte (Kinder) auf Bildern zu erkennen, zu kategorisieren
und entsprechend in eigene Ordner zu sortieren. Am Ende lässt sich eine Datei mit den Bildnamen, deren Pfaden und deren
Kategorie im gewünschten Format
ausgeben.

Es werden alle notwendigen Schritte beschrieben. Zum Beispiel wie die notwendige Software installiert
wird und welche Packages/Abhängigkeiten verwendet werden.

Darauf folgend wird erklärt wie der Workflow aussieht um dieses Model zu verwenden und es soll möglich sein die KI mit
neuen Labels zu trainieren.

## Verwendete Software

- Ein Editor wie [DataSpell](https://www.jetbrains.com/de-de/dataspell/)
  /[Pycharm](https://www.jetbrains.com/de-de/pycharm/) oder einfach [Jupyter Notebook](https://jupyter.org/install)
- [Label Studio](https://labelstud.io) für das Labeling und Evaluieren des Models
- [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Setup

### Installation Tensorflow object detection api

[Offizielles Setup](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation)

Zunächst muss der PC eingerichtet werden. Dazu gibt es das offizielle Tutorial welches unter MacOS leider nicht richtig
funktioniert, jedoch für Windows und Linux geschrieben ist. Dennoch sollten unter MacOS die Schritte ausgeführt werden.
Bei **Problemen** folgendes versuchen:

- miniconda3 installieren
- conda env erstellen
- das erstellte env aktivieren
- Befehle aus INSTALLATION.md ausführen (werden benötigt damit folges funktioniert)
- [Downloading the TensorFlow Model Garden](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#downloading-the-tensorflow-model-garden)
- [Install the Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-the-object-detection-api)
  bei Fehlern bzgl. COCO
  einfach [COCO API installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#coco-api-installation)
  ausprobieren
- [Test your Installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#test-your-installation)

### Installation label-studio

...

### Pfad für zu predictende Bilder

Unter "object_detection/images/" werden die Bilder abgelegt, welche von der KI genutzt werden sollen.
Diese werden nicht ins Projekt hochgeladen und nur lokal vom Code verwaltet.