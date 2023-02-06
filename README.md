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

## Ein eigenes Model Trainieren

>In diesem Abschnitt werden die Inhalte der folgenden Ordner erläutert.
> Ein weiters Modell kann mithilfe dieser Erläuterung nachgebaut werden.
>```
> workspace
> └── model1
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
>### model1 
> Der Ordner model1 soll unser Trainingsordner sein, der alle Dateien für das Training unseres Modells enthält.
> Es ist ratsam, jedes Mal, wenn wir mit einem anderen Datensatz trainieren wollen, einen separaten Trainingsordner zu erstellen.
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
> Diese Dateien werde verwendet, um den Datensatz an die KI zu übertragen. Sie sind für das Trainieren des Models notwendig.
> Um diese Dateien zu erstellen Führen Sie die folgenden Punkte aus:
>> - Exportieren Sie den annotierten Datensatz vom label-Studio in "Pascal VOC XML" Format
>> - Entpacken Sie die heruntergeladene Datei und kopieren Sie alle resultierenden Bilder und XML Dateien in den Ordner "model1/images/exported_datasets"
>> - Exportieren Sie den Datensatz in Label-Studio auch als "JSON" File und bewegen Sie den JSON nach "ki-anwendung_object-detection/Scripts"
>> - Führen Sie die Datei "ki-anwendung_object-detection/Scripts"/split_datasets_in_test_and_train.ipynb" aus.         Als Resultat werden die Bilder und xml Datein unter "ki-anwendung_objekt-detection/Tensorflow/workspace/model1/images/exported_datasets"
>> auf die zwei ordner "train" und "test" verteilt. 
>> - Öffnen Sie den Notebook "ki-anwendung_object-detection/training-manager.ipynp" und führen Sie die Zelle "Create TF records" aus.
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
>> - Ein tensorflow Modell herunterladen (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
>> - Nachdem Sie die *.tar.gz-Datei heruntergeladen haben, öffnen Sie sie mit einem Dekomprimierungsprogramm Ihrer Wahl
>> (z. B. 7zip, WinZIP usw.). Öffnen Sie dann den *.tar-Ordner, den Sie sehen, wenn der komprimierte Ordner geöffnet 
>> wird, und extrahieren Sie seinen Inhalt in den Ordner model1/pre-trained-models.
>> - Da wir das "CenterNet HourGlass104 512x512" Modell heruntergeladen haben, sollte unser training_demo Verzeichnis 
>> nun wie folgt aussehen: 
>>```
>> model1/
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
>>  Um einen neuen Trainingsauftrag zu erstellen, gehen Sie folgendermaßen vor:
>>   - Unter model1/models erstellen Sie ein
>>      neues Verzeichnis mit dem Namen centernet_hg104_512x512_coco17_tpu-8 
>>   - kopieren Sie die Datei model1/pre-trained-models/centernet_hg104_512x512_coco17_tpu-8/pipeline.config in das neu erstellte
>>      Verzeichnis. 
>>   - Unser model1/models-Verzeichnis sollte nun wie folgt aussehen: 
>>    ```     
>>       ── models
>>          └── centernet_hg104_512x512_coco17_tpu-8
>>              └── pipeline.config
>>    ```
>>  
>> - pipline.config:     
>>  In dieser Datei werden die Pfade und Parameter des zu trainierenden Modells eingestellt.    
>>  Jedes vortrainierte Modelle hat eigene pipline.config.    
>>  Hier sind die Änderungen, die in diesem Projekt an der 
>>  "model1/models/centernet_hg104_512x512_coco17_tpu-8/pipeline.config" vorgenommen wurden:
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
>### exporter_main_v2.py
> Bei der Ausführung dieser python Datei mit den richtigen Parametern, wird ein trainiertes Modell exportiert
>### generate_tfrecord.py
> Bei der Ausführung dieser python Datei wird der Datensatz (images, xml) zu tensorflow.records Konvertiert. 
>### model_main_tf2.py
> Diese Datei ist die main Datei für das Trainieren eines Modells.
>## Starten des Trainings
>Sobald der Ordner "model1" wie beschrieben erstellt wurde, kann das Weiter-Training gestartet werden.
>Öffnen Sie die Datei "training-manager.ipynb" und führen Sie die entsprechenden Zellen aus.


 
 


