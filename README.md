# Workflow für anwendung und erweiterung eines Object Detection Models

## Ordnerstruktur

```
.
├── INSTALLATION.md
├── README.md
└── object_detection
    ├── images
    │   └── example.png
    │   ├── all_images
    │   ├── detected_persons
    │   └── not_relevant_label
    ├── saved_modles
    │   └── example_model
    ├── plot_object_detection_saved_model.ipynb
    └── Extract_Person_Images.ipynb
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

Darauf folgend wird erklärt, wie der Workflow aussieht, um dieses Model zu verwenden. Es soll möglich sein die KI mit
neuen Labels zu trainieren.

## Verwendete Software

- Ein Editor wie [DataSpell](https://www.jetbrains.com/de-de/dataspell/)
  /[Pycharm](https://www.jetbrains.com/de-de/pycharm/) oder einfach [Jupyter Notebook](https://jupyter.org/install)
- [Label Studio](https://labelstud.io) für das Labeling und Evaluieren des Models
- [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Setup

Um inferences auf Bilder auszuführen, ein Modell zu trainieren und zu evaluieren werden bestimmte Python Packages benötigt. Dies beschreiben die folgenden Schritte.

### 1. Installation von Git LFS für das Modell

1. git lfs
   installieren [docs.github.com](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
2. ```git lfs pull``` ausführen

### 2. Installation von miniconda 

- miniconda3 installieren

### 3. Installation von Tensorflow object detection api

Zunächst muss der PC eingerichtet werden. Dazu gibt es das [offizielle Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation) welches unter MacOS leider nicht richtig
funktioniert, weil es für Windows und Linux geschrieben ist. Eine einfachere Variante, welche unter allen Betriebssystemen funktionieren sollte, ist unser Vorgehen, welches wie folgt aussieht und NVIDIA GPUs unterstützt.

```
conda create -n tf python=3.9
conda activate tf
pip install --ignore-installed --upgrade tensorflow
# Clone the tensorflow models repository
git clone https://github.com/tensorflow/models.git
cd models/research/

protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
Nun kann die installation getestet werden mit dem Befehl:   
`python3 object_detection/builders/model_builder_tf2_test.py`
```
# Für NVIDIA GPU Support
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install --upgrade tensorrt
```
Um nun zu prüfen ob die GPU erkannt wird, kann dieser Befehl genutzt werden.  
`python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### 4. (optional) Installation von label-studio zum annotieren von Bildern

...

## Pfad für zu predictende Bilder und Trainingsmodel

Unter "object_detection/images/" werden die Bilder abgelegt, welche von der KI genutzt werden sollen.
Diese werden nicht ins Projekt hochgeladen und nur lokal vom Code verwaltet.

Unter "object_detection/saved_models/" werden die Modelle gespeichert,
die zum Vorsortieren der Bilder bzw. Anlernen der KI gespeichert. Diese werden nicht ins Projekt hochgeladen und nur
lokal vom Code verwaltet.

## Skripte

Zu finden sind die Skripte im Unterordner `Scripts/`

Extract_Person_Images.ipynb:

> wird einmalig verwendet, um die Quell-Bilder vorerst vorzusortieren. Das Skript wurde
> so angepasst, dass es auf jedem Rechner läuft, wenn die Ordnerstruktur eingehalten wurde.
> Unter dem Ordner all_images sollten alle Quell-Bilder sein. Unter saved_models soll das Model "
> centernet_hg104_1024x1024_coco17_tpu-32"
> mit seinem variabels-Ordner installiert sein.
> Das Skript verarbeitet jedes Quell-Bild vom Ordner image/all_images. Durch das centernet
> Model werden menschen auf dem Bild ggf. erkannt und das Bild wird in den Ordner "images/detected_persons"
> kopiert. Wenn keine Menschen auf dem Bild zu erkennen sind, wird das Bild in den Ordner "images/not_relevant_label"
> kopiert

plot_object_detection_saved_model.ipynb

> Beschreibung
