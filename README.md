# Anwendung und erweiterung eines Object Detection Models mit Tensorflow

Das Projekt wurde erstellt als Kooperation zwischen der FH Kiel und dem Badischen Landesmuseum Karlsruhe im Rahmen von „Creative User Empowerment".

## Zusammenfassung

In diesem Projekt wird mittels Transferlearning ein etablierstes Object Detection Model verwendet und mit eigenen
Bildern erweitert. Dieses neue Model soll dazu dienen Objekte (Kinder) auf Bildern zu erkennen und zu kategorisieren. Am Ende lässt sich eine Datei mit den Bildnamen, deren Pfaden und deren
Kategorie im gewünschten Format ausgeben (z.B. CSV).

Es werden alle notwendigen Schritte beschrieben. Zum Beispiel wie die notwendige Software installiert
wird und welche Packages/Abhängigkeiten verwendet werden.

Darauf folgend wird erklärt, wie der Workflow aussieht, um dieses Model zu verwenden. Es soll möglich sein die KI mit
neuen Labels zu trainieren. Dafür werden die Bilder mit label studio annotiert.

---

## Skripte

Zu finden sind die Skripte im Unterordner `Scripts/`.

#### find_problems_in_dataset_export.ipynb:

Dieses Skript bietet die Möglichkeit, annotierte Datensätze nach Fehlern zu durchsuchen und alle Labels/Klassen zu
finden.
Hierfür wird nach dem annotieren in label-studio ein JSON exportiert und dieses im Ordner `Scripts/` abgelegt.

#### split_datasets_in_test_and_train.ipynb:

Hiermit werden anhand des exportierten JSON die exportierten Bilder im PASCAL VOC Format zu Test- und
Trainings-datensätzen getrennt und in die richtigen Ordner verschoben. Diese werden später verwendet um daraus
kompatible Datensätze für das Modell zu generieren (.record).

#### training_manager.ipynb:

Zum trainieren, evaluieren und exportieren eines eigenen Modells sind hier die Pythonskripte mit den notwendigen
Parametern und Pfaden, zum einfachen nuzen hinterlegt.

#### generate_csv.ipynb:

Um die Vorhersagen auf Bilder durch ein Modell zu speichern, wird eine .csv Datei erstellt und mit den Namen des
jeweiligen Bildes und der Vorhersage des Modells befüllt.

---

## Verwendete Software

- Ein Editor wie [DataSpell](https://www.jetbrains.com/de-de/dataspell/)
  /[Pycharm](https://www.jetbrains.com/de-de/pycharm/) oder einfach [Jupyter Notebook](https://jupyter.org/install)
- Jupyter Notebook Plugin falls IDE verwendet wird
- [Label Studio](https://labelstud.io) für das Labeling und Evaluieren des Models
- [miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## Setup

Um inferences auf Bilder auszuführen, ein Modell zu trainieren und zu evaluieren werden bestimmte Python Packages
benötigt. Dies beschreiben die folgenden Schritte.

### 1. Projekt clonen in den Pfad deiner Wahl

- `git clone ....`

### 2. Installation von Git LFS für das Modell

- git lfs
  installieren: [docs.github.com](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

### 3. Installation und einrichtung von miniconda

- miniconda3 installieren: [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)

``` bash
conda create -n tf python=3.9 
conda activate tf
```

### 4. Installation von DataSpell

- [jetbrains.com/de-de](https://www.jetbrains.com/de-de/dataspell/) installieren
- Umgebung einrichten (Darauf achten, dass das richtige conda environment **tf** ausgewählt
  wird) [Anleitung](https://www.jetbrains.com/de-de/dataspell/quick-start/)
- Lokalen Ordner an den Workspace anhängen [Anleitung](https://www.jetbrains.com/de-de/dataspell/quick-start/)

### 5. Installation von Tensorflow object detection api

Zunächst muss das conda environment für object detection eingerichtet werden. Dazu gibt es
das [offizielle Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation)
welches für MacOS leider nicht richtig
beschrieben wird. Dafür für Windows und Linux. Bei Problemen mit den Befehlen das offizielle Tutorial
lesen.
Eine einfachere Variante, welche unter allen
Betriebssystemen funktionieren sollte, ist unser Vorgehen, welches wie folgt aussieht und NVIDIA GPUs unterstützt.

```
# Navigiere in das Projekt
cd /Pfad/Zum/Geklonten/Projekt
git lfs pull
conda activate tf
pip install --ignore-installed --upgrade tensorflow
cd TensorFlow
git clone https://github.com/tensorflow/models.git
cd models/research
# Protoc installieren wie im offiziellen Tutorial. Link oben zu finden.
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

### 6. (optional) Installation von label-studio zum annotieren von Bildern

label-studio kann über zwei Wege genutzt werden:

1. docker-compose.yml starten, dann werden alle Daten im Ordner `labelstudio-data` verwaltet
2. label-studio mit pip im conda environment installieren & starten, dann werden alle Daten im conda environmend
   verwaltet

Installation mit conda: [labelstud.io](https://labelstud.io/guide/install.html)

### 7. (optional) Machine learning backend zum prelabeln

Pre-labeln bedeutet, eine Menge an Bilder von einem vortrainierten Model vor annotieren lassen.
Dies beschleunigt die Arbeit und zeigt die genauigkeit des verwendeten Modells. 
Zum prelabeln wird ein Modell benötigt, die Anleitung hierzu findet man im nächsten Abschnitt *Ein eigenes Modell
tranieren* unter *pre-trained-models*.

Das Machine Learing Backend kann über zwei Wege gestartet werden: 
1. docker-compose.yml starten.
2. label-studio-ml-backend mit pip im conda environment installieren & starten [How to set up machine learning with  label-Studio](https://labelstud.io/guide/ml.html#How-to-set-up-machine-learning-with-Label-Studio)

Nach dem Starten des Ml Backends sollten die wunsch Labels in Label-Studio unter Setting/labeling Interface gesetzte werden.         
Das Anbinden des ML Backends an Label-Studio Frontend erfolgt unter 
Label-Studio --> Setting --> Machine Learing --> Add Model --> URL: ( z.B. http://backend.develop:9090 wenn das Backend
mit unserem Docker file gestartet haben)  
für weitere Informationen besuchen Sie das offizielle guid [Add ML Backend in Label Studio](https://labelstud.io/guide/ml.html#Add-an-ML-backend-using-the-Label-Studio-UI)   




---

## Inference/prediction auf Bilder ausführen

Unser trainiertes Modell erkennt Kinder, Frauen und Männer.
Hierfür haben wir ein Skript, welches eine CSV generiert mit den Namen des Bilder

### Pfad für zu predictende Bilder und Trainingsmodel

```
.
├── README.md
├── Scripts
├── TensorFlow
├── images
│   └── all_images <--
└── ...
```

Unter "images/all_images" werden die Bilder abgelegt, welche von der KI inferenced/predicted werden sollen.

Unter "object_detection/saved_models/" werden die Modelle gespeichert,
die zum Vorsortieren der Bilder bzw. Anlernen der KI gespeichert. Diese werden nicht ins Projekt hochgeladen und nur
lokal vom Code verwaltet.

---

## Datensatz generieren

### Annotieren von Bildern mit label-studio

Um ein eigenes Modell weitertrainieren zu können. müssen eigene Datensätze erstellt werden. Hierfür kann label-studio
genutzt werden.
Der offizielle Guide zum annotieren: [labelstud.io](https://labelstud.io/guide/)

### Bereinigen von Fehlern in den annotierten Bildern

Um zu prüfen, ob nur die benötigten Labels/Klassen in den annotierten Bildern zu finden sind, verwenden wir das
Skript `find_problems_in_dataset_export.ipynb`

Zunächst muss in label-studio ein JSON exportiert werden. Dies enthält
alle namen und sonstige Informationen zu den Bildern welche zuvor annotiert wurden.

Die exportierte Datei `project-x-at-xxxx-xx-xx-xx-xx-xxxxxxxx.json` wird nach dem export im Downloadsordner des
Computers liegen.
Diese Datei dann in den Ordner `Scripts/` des Projekts legen und im Skript `find_problems_in_dataset_export.ipynb` den
Namen in der Variable `path_to_json` ersetzen.

Dann muss die Zelle ausgeführt werden. Der Output sollte in etwa so aussehen (nach dem bereinigen):

```
Label names:  {'child', 'man', 'person', 'woman'}
Pictures with unnamed labels:  0
PictureIds with unnamed labels:  []
```

Zu sehen sind alle gefundenen Labels, Anzahl der Bilder welche Annotationen ohne Namen enthalten und die entsprechenden
IDs dazu. Diese IDs entsprechen den IDs welche in label-studio angezeigt werden. Anhand dieser kann in label-studio
manuell jedes Bild von nolabels bereinigt werden.

### Exportieren der Datensätze in PASCAL
In Label Studio gibt es die Option, den annotierten Datensatz durch den Export-Button in verschiedene Formate weiterzugeben. 
Exportieren Sie den annotieren Datensatz aus Label Studio im Pascal VOC XML-Format.          
Nach dem Herunterladen der Datei entpacken Sie den Ordner, um die resultierenden Bilder und XML-Dateien sehen zu können.

---

## Ein eigenes Modell Trainieren

> In diesem Abschnitt werden die Inhalte der folgenden Ordner erläutert.
> Ein weiters Modell kann mithilfe dieser Erläuterung nachgebaut werden.
>```
> workspace
> └── project_museum
>   ├── annotations
>   │         ├── label_map.pbtxt
>   │         ├── test.record
>   │         └── train.record
>   ├── exported-models
>   │   └── myModel
>   │       ├── pipeline.config
>   │       ├── checkpoint
>   │       └── saved_model
>   │           ├── assets
>   │           └── variables
>   ├── images
>   │   ├── exported_datasets
>   │   ├── test
>   │   └── train
>   ├── models
>   │   ├── centernet_hg104_512x512_coco17_tpu-8
>   │       ├── pipeline.config
>   │       ├── eval
>   │       ├── train
>   │       └── checkpoint
>   ├── pre-trained-models
>   │   └── centernet_hg104_512x512_coco17_tpu-8
>   │       ├── pipeline.config
>   │       ├── checkpoint
>   │       └── saved_model
>   │           └── variables
>   ├── exporter_main_v2.py
>   ├── generate_tfrecord.py
>   └── model_main_tf2.py
>```
>### project_museum
> Der Ordner project_museum soll unser Trainingsordner sein, der alle Dateien für das Training unseres Modells enthält.
> Es ist ratsam, jedes Mal, wenn wir mit einem anderen Datensatz trainieren wollen, einen separaten Trainingsordner zu
> erstellen.
> Die typische Struktur für Trainingsordner kann, wie unten beschrieben, nachgemacht werden.
>### annotations
>```
> annotations              
>       ├── label_map.pbtxt 
>       ├── test.record    
>       └── train.record
> ```
> - label_map.pbtxt:
    > Tensor Flow benötigt eine Label-Map, die jedes der verwendeten Labels auf einen ganzzahligen Wert abbildet.
    > Die Datei kann mit einem Text Editor erstellt werden.
    > Unten wird ein Beispiel-Label-Map (z.B. label_map.pbtxt) gezeigt, unter der Annahme, dass der Datensatz 4 Labels
    > enthält: person, man, woman, child:
>```
>item {
>id: 1
>name: 'person'
>}
>
>item {
>id: 2
>name: 'man'
>}
>
>item {
>id: 3
>name: 'woman'
>}
>...
>```
> - test.record, train.record:
    > Diese Dateien werde verwendet, um den Datensatz an die KI zu übertragen. Sie sind für das Trainieren des Models
    notwendig.
    > Um diese Dateien zu erstellen Führen Sie die folgenden Punkte aus:
>> - Exportieren Sie den annotierten Datensatz vom label-Studio in "Pascal VOC XML" Format
>> - Entpacken Sie die heruntergeladene Datei und kopieren Sie alle resultierenden Bilder und XML Dateien in den
     Ordner "project_museum/images/exported_datasets"
>> - Exportieren Sie den Datensatz in Label-Studio auch als "JSON" File und bewegen Sie den JSON nach "
     ki-anwendung_object-detection/Scripts"
>> - Führen Sie die Datei "ki-anwendung_object-detection/Scripts"/split_datasets_in_test_and_train.ipynb" aus. Als
     Resultat werden die Bilder und xml Datein unter "
     ki-anwendung_objekt-detection/Tensorflow/workspace/project_museum/images/exported_datasets"
     > > auf die zwei ordner "train" und "test" verteilt.
>> - Öffnen Sie den Notebook "ki-anwendung_object-detection/training-manager.ipynp" und führen Sie die Zelle "Create TF
     records" aus.
>### exported-models
> In diesem Ordner werden die exportierten Versionen unserer trainierten Modelle gespeichert. bsp. myModel          
> Um ein trainiertes Modell zu exportieren Führen Sie die folgenden Schritte aus:
>> - Öffnen Sie den Notebook "ki-anwendung_object-detection/training-manager.ipynp"
>> - Tragen Sie den Namen des Modellordners z.B."centernet_hg104_512x512_coco17_tpu-8" in die erste zeile ein.
>> - Führen Sie die Zelle "Export the Model" aus
>### images
> Dieser Ordner enthält eine Kopie aller Bilder in unserem Datensatz sowie die entsprechenden .xml-Dateien,
> die für jedes Bild erstellt werden, sobald Label-Studio zum Annotieren der Objekte verwendet wird.
>### pre-trained-models
> Dieser Ordner enthält die heruntergeladenen vortrainierten Modelle, die als Startpunkt für unsere
> Trainingsaufgaben verwendet werden sollen.   
> Um ein Modell weiter zu trainieren Führen Sie die folgenden Schritte aus:
>> - Ein tensorflow Modell
     herunterladen (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
>> - Nachdem Sie die *.tar.gz-Datei heruntergeladen haben, öffnen Sie sie mit einem Dekomprimierungsprogramm Ihrer Wahl
     > > (z. B. 7zip, WinZIP usw.). Öffnen Sie dann den *.tar-Ordner, den Sie sehen, wenn der komprimierte Ordner
     geöffnet
     > > wird, und extrahieren Sie seinen Inhalt in den Ordner project_museum/pre-trained-models.
>> - Da wir das "CenterNet HourGlass104 512x512" Modell heruntergeladen haben, sollte unser training_demo Verzeichnis
     > > nun wie folgt aussehen:
>>```
>> project_museum/
>>├─ pre-trained-models/
>>│  ├─ centernet_hg104_512x512_coco17_tpu-8/
>>│  │  ├─ checkpoint/
>>│  │  ├─ saved_model/
>>│  │  └─ pipeline.config
>>│  └─ ...
>>└─ ...
>>```
>### models
> Dieser Ordner enthält einen Unterordner für jeden Trainingsauftrag.
> Jeder Unterordner enthält die Trainings-Pipeline-Konfigurationsdatei pipeline.config
> sowie, später nach dem Training, alle Dateien,
> die während des Trainings und der Auswertung unseres Modells erzeugt werden.
>
>> - Neuer Trainingsauftrag:         
     > > Um einen neuen Trainingsauftrag zu erstellen, gehen Sie folgendermaßen vor:
     >>

- Unter project_museum/models erstellen Sie ein
  > > neues Verzeichnis mit dem Namen centernet_hg104_512x512_coco17_tpu-8

> > - kopieren Sie die Datei project_museum/pre-trained-models/centernet_hg104_512x512_coco17_tpu-8/pipeline.config in das neu
      erstellte
      > > Verzeichnis.
      >>

- Unser project_museum/models-Verzeichnis sollte nun wie folgt aussehen:
  > >    ```     
  > > ── models
  > > └── centernet_hg104_512x512_coco17_tpu-8
  > > └── pipeline.config
  >>    ```

> >
>> - pipline.config:     
     > > In dieser Datei werden die Pfade und Parameter des zu trainierenden Modells eingestellt.    
     > > Jedes vortrainierte Modelle hat eigene pipline.config.    
     > > Hier sind die Änderungen, die in diesem Projekt an der
     > > "project_museum/models/centernet_hg104_512x512_coco17_tpu-8/pipeline.config" vorgenommen wurden:
>> ```
>>     Zeile 3      num_classes: 4            
>>     Zeile 44     batch_size: 4           
>>     Zeile 100    fine_tune_checkpoint: "pre-trained-models/centernet_hg104_512x512_coco17_tpu-8/checkpoint/ckpt-0"             
>>     Zeile 101    num_steps: 17000    #je nach Bildanzahl einstellen ((1 step = 1 Bild) wenn batch_size = 1)((1 step = 4 Bild) wenn batch_size = 4)               
>>     Zeile 104    fine_tune_checkpoint_type: "detection"             
>>     Zeile 105    fine_tune_checkpoint_version: V2
>>     Zeile 108    label_map_path: "annotations/label_map.pbtxt"             
>>     Zeile 110    input_path: "annotations/train.record"             
>>     Zeile 114    metrics_set: "coco_detection_metrics"             
>>     Zeile 115    use_moving_averages: false             
>>     Zeile 116    batch_size: 1             
>>     Zeile 119    label_map_path: "annotations/label_map.pbtxt"
>>     Zeile 123    input_path: "annotations/test.record"
>> ```
>## Training starten
>Sobald der Ordner "project_museum" wie beschrieben erstellt wurde, kann das Weiter-Training gestartet werden.
> Öffnen Sie die Datei "training-manager.ipynb" und führen Sie die entsprechenden Zellen aus.
>### exporter_main_v2.py
> Bei der Ausführung dieser python Datei mit den richtigen Parametern, wird ein trainiertes Modell exportiert
>### generate_tfrecord.py
> Bei der Ausführung dieser python Datei wird der Datensatz (images, xml) zu tensorflow.records Konvertiert.
>### model_main_tf2.py
> Diese Datei ist die main Datei für das Trainieren eines Modells.

### Starten des Trainings

Sobald der Ordner "model1" wie beschrieben erstellt wurde, kann das Weiter-Training gestartet werden.
Öffnen Sie die Datei "training-manager.ipynb" und führen Sie die entsprechenden Zellen aus.


 
 


