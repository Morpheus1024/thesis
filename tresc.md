# Praca inżynierska

## 1. Wstęp

### 1.1 Cel Pracy

W niniejszej pracy przedstawiono proces tworzenia i operowania na trójwymiarowej mapie semantycznej na podstawie danych pochodzących z kamery RGPB-D Intel RealSense D435.
Używając dostępnych bibliotek udostępnionych przez producenta odczytano dane z kamery, które posłużyły jako podstawa do tworzenia mapy semantycznej. Następnie, przez odpowiednią obróbkę danych jak i wykorzystanie dostępnych modeli dokonano segmentacji obiektów widzianych przez kamerę. Na tej podstawie stworzono trójwymiarową mapę semantyczną widzianego obrazu.

### 1.2  Definicja segmentacji semantycznej

Segmentacja semantyczna jest podzadaniem segmentacji panoptycznej, którą definiuje się jako przypisanie każdemu pixelowi analizowanego obrazu etykiety semantycznej oraz identyfikacji każdej z instancji występującej na obrazie. Etykiety są zazwyczaj dzielone na te opisujące na obiektach policzalnych - ang. things - np. osoby, samochody, drzewa, oraz obiektach niepoliczalnych i amorficznych - ang. stuff - takie jak niebo, droga. [przypis 2. rozdział 1]. Operacje segmentacji wykonywane tych drugich są określane mianem segmentacji semantycznej [przypis 1. rozdział 1.].
W kontekście 3D, danymi poddawanymi analizie nie jest już samo zdjecie, ale również informacja o głebi odczytywana między innymi z chmury punktów.

### 1.3 Trójwymiarowa mapa semantyczna

Mapa semantyczna jest graficzną reprezentacją informacji w danej dziedzinie lub tematyce wraz ze wzajemnymi relacjami pomiędzy poszczególnymi pojęciami. W kontekście trójwymiarowości oraz wizji komputerowej jest sposobem na reprezentowanie środowiska łączace informacje o przestrzeni, w tym geometrii obiektów, wraz z semantycznymi etykietami. Pozwala to na przekazanie przez komputer informacji o położeniu obiektów oraz ich znaczeniu/przynależności/reprezentacji.
W praktyce, mapa jest tworzona poprzez połączenie informacji pozyskanych z chmury punktów tj. kolorów oraz głębi wraz z danymi otrzymanych w wyniku segmentacji semantycznej - etykiet przypisanych do każdego punktu chmury.
Trójwymiarowe mapy semantyczne znajdują coraz większe zastosowanie w robotyce, szczególnie w rozwiązaniach przemysłowych z dynamicznie zmienającym się otoczeniem robota. Stosowane są również w rzeczywistości rozszerzonej i wirtualnych w celu tworzenia symulacji i środowisk wirtualnych np. na potrzeby medyczne. Są również kluczowe w autonomicznie poruszających się pojazdach jak i w przemyśle np. w celu kontroli efektów pracy zautomatyzowanej linii produkcujnej.

### 1.4 Opis kamery i biblioteki

Kamera wykorzystana w niniejszej pracy to Intel® RealSense™ Depth Camera D435. Wyposażona jest ona w klasyczny obiektyw RGB jak i oprzyrządowanie do odczytania informacji o głębi obrazu. Wykorzystuje do tego rzutnik punktów widocznych w podczerwieni, których pozycja jest określana przez stereoskopowe czujniki podczerwieni. Producent określa odległość roboczą przyrządu od 30 cm do 3 m. [przypis 3. specyfikacja techniczna]. W przeciwieństwie do kamer wyposarzonych w dwa obiektywy RGB, kamera RGBD lepiej sprawuje się w zamkniętych pomieszczeniach i na mniejszych dystansach.
Producent dostarcza również bibliotekę librealsense, która pozwala na zmianę domyślnych parametrów kamery. W niniejszej pracy została wykorzystana jej odmiana napisana w języku Python - pyrealsense. Jest wyposażona w gotowe funkcje do odczytu obrazu RGB i głębi w danych rozdzielczościach, funkcję wyrównywania obu obrazów, czy prostego zapisywania chmury punktów 3d do pliku o rozszerzeniu .ply.

## 2. Przegląd literatury

Z roku na rok powstają coraz nowsze sposoby i podejścia do segmentacji zdjęć i filmów. Poddawane są one testom wydajnościowym na otwartoźródłowych, powszechnie uznanych wśród społeczności zbiorach danych. Trudno nadążyć za najnowyszymi i najwydajniejszymi systemami z uwagi na szybkość zmian jakie zachodzą w tej dziedzinie.
W tym rozdziale zostaną opisane jedne z najpopularniejszych używanych systemy segmentacji, które są aktywnie używane w tym celu.

(prawdopodobnie do wywalenia) Do najpopularniejszych frameworków segmentacji panoptycznej można zaliczyć transformers [przypis 8]. Wybrano go z uwagi na dostępność, wsparcie społeczności i popularność na repozytorium Github oraz ciągły rozwój w celu osiągnięcia coraz lepszych wyników wydajności.


### 2.1 Modele segmentacji semnatycznej

#### 2.1.1 Transformers

Transformers oddaje w ręce użytkownika API, które pozwalają na używanie już wytrenowanych modelu. Framework jest szeroko stosowany w dziedzinach związanych z NLP, audio czy chociażby z wizją komputerową.
Transformers został zaprojektowany by jak najlepiej odwzorować modele potokowe tzn. dokonać wstępnej obróbki danych, poddać je działaniu modelu i dokonać predykcji. Każdy z modeli został zdefiniowany poprzez trzy bloki stanowiące rdzeń działania całego frameworka: blok tokenizacji danych, blok transformacji (od którego wzięła się nazwa frameworka), oraz z bloku głowy/głów.
Blok tokenizacji - również nazywany tokenizerem - jest odpowiedzialny za nadawanie stokenizowanych klas, które są niezbędne do pracy każdego modelu. Klasy mogą być już predefiniowane, ale mogą również zostać dodane przez użytkownika. Przechowuje on listę mapującą token do indeksu.
Blok transformacji ma za zadanie wykonywać zadanie modelu np. generowanie, rozumowanie na podstawie klas powstałych w wyniku tokenizacji w poprzednim bloku. Architektury modeli zostały w taki sposób dobrane, by była możliwość łatwego podmieniania ich bloku transformacji.
Blok głów jest odpowiedzialny za dostosowanie danych otrzymanych z bloku transformacji do danych wyjściowych dostosowanych do danego zadania np. współrzędne bounding boxa. Dodaje on do klasy bazowej warstwę wyjścia oraz funkcję straty. Niektóre bloki obsługują również dodatkowe funkcje takie jak próbkowanie w celu wykonania powierzonego im zadania.
Celem autorów Transformers było stworzenie hubu wytrenowanych modeli w celu ułatwienia dostępu do nich oraz łatwego aplikowania ich do projektów użytkowników. W 2020 roku hub oferował ponad 2000 modeli, w tym BERT i GPT-2. Na czas pisania tej pracy modeli jest ponad 660000.
Modele dostępne w Transformers można zainstalować poprzez instalację biblioteki PyTorch, Tensorflow oraz Flex jak i poprzez bezpośrednie pobranie ze strony projektu na Githubie.

#### 2.1.2 YOLO

brak papierów o yolov8.

[dostępne papiery](https://arxiv.org/pdf/2304.00501v1)

### 2.2 Modele segmentacji semantycznej 3D

Oprócz klasycznych modeli zajmujących się segmentacją obrazów trzeba również wyróżnić modele specjalizujące się segmentacją chmur punktów, zawierających oprócz informacji o kolorach również dane o głębi zarejestrowanego obrazu. Są one o wiele mniej popularne z uwagi na znacznie rzadsze koszystanie z obrazów RGBD - więcej osób posiada obiektywy do wykonywania zdjęć 2D, w szczególności smartfony, niż sprzęt pozwalający na pozyskanie danych o głębi zdjęcia. Postanowiono na opisanie zasady działania modeli, które odznaczyły się w testach wydajnościowych w momencie pisania tej pracy na zbiorach SemanticKITTI, ScanNet200, ScanNet++ - PTv3 + PTT, oraz S3DIS - OneFormer3D.

#### 2.2.1 Point Transformer V3

Najczęściej wskazywanym modelem na stonie gromadzącej modele State of the Art związanych z segemntacją semantyczna 3d [Przypis 7] jest Point Transformer V3. Jest on rozwinięciem swojego poprzednika - PTv2. Autorzy chwalą się, że jest szybszy, a zarazem elastyczny i zachowuje prostotę.
Dane z chmury punktów przed podaniem ich do modelu są normalizowane. Stosowana jest normalizacja warstw zamiast normalizaji batchy danych. Następnie dane są poddawane gridpoolinowi, co jest cechą zachowaną z poprzedniej iteracji modelu. Dodano warstwę konwolucyjną, która pozwala na zamianę kodowania pozycyjnego punktów na kodowanie xCPE (enhanced Conditional Position Embedding), co dało lepszą dokładność danych wynikowych przy minimalnie większym nakładzie obliczeniowym.

#### 2.2.2 OneFormer3D

OneFormer3D jest określany przez twórców jako model do ujednoliconej segmentacji chmury punktów. Chwalą się, że jako pierwszy model łączy backbone z U-Net z dekoderem tranformer, co daje wzrot wydajności pod kątem czasu operacji i bardziej zoptymalizowanego użycia pamięci.
Zasada działania modelu wygląda w następujący sposób:
Na podstawie koloru oraz położenia danego punktu w przestrzeni ekstraktowane są cechy chmury przez 3D U-Net. Następnie chechy punktów są uśredniane za pomocą "flexible poolingu", a same punkty są łączone w nazywane przez autorów superpunkty. Cechy tych punktów są przekazywane do dekodera o 6 warstwach. W celu trenowania sieci można również podać dodatkowe zapytania semantyczne i instancyjne. Dekoder zwraca zestaw kernel segmentowanych masek obiektów ze zdjęcia. Mogą to być maski będące wynikiem segmentacji instancyjnej, semantycznej czy panoptycznej.

## 3. Metody tworzenia trójwymiarowej mapy semantycznej i jej wizualizacji

### 3.1 Opis metod tworzenia trójwymiarowej mapy semantycznej

Tworzenie trójwymiarowej mapy semantycznej opiera się na połączeniu informacji pochodzących z segmegmentowanego obrazu 2D z danymi o głębi scenerii. Aktualnie opracowano wiele metod, które pozwalają na osiągnięcie różnych efektów. Wyróżnić można m.in. prostą projekcję, metody iteracyjne czy geometryczne.

#### 3.1.1 Projekcja prosta

Projekcja prosta jest zdecydowanie najłatwiejszą metodą do zaimplementowania. Polega na odwzorowaniu pikseli segmentowanego obrazu bądź z maski pochodzącej z segmentacji na odpowiadające im punkty chmury głębi.
Metoda cechuje się prostotą, szybkościa implementacji oraz intuicyjnością. Nie mniej, zawiera wady w postaci braku uwzględnienia szczegółów geometrii obiektów i braku uwzględnienia niedoskonałości danych, w szczególności szumów pochodzących z odczytu głębi.

#### 3.1.2 Metody iteracyjne

Rozwinięciem metody projekcji prostej jest iteracyjne udoskonalanie dopasowania maski semantycznej do chmury punktów. Polega to na liczeniu wektorów normalnych dla każdego punktu obiektu i dopasowaniu granic poprzez przesuwanie punktów wzdłuż wektorów. Pozwala to na osiągnięcie większej dokładności w stosunku do projekcji prostej przy jednoczesnej większej złożoności obliczeniowej i większej wrażliwości na wskazane przez użytkownika parametry algorytmu. Przykładowe metody to  Active Contours, Level Sets, Deformable Models czy Markow Random Fields.

#### 3.1.3 Metody geometryczne

Odmienne podejście jest stosowane w metodach geometrycznych. W nich podstawą działania algorytmu są uproszczone modele geometryczne zwane prymitywami. Na początku algorytm stosuje się segmentację regionów na podstawie maski segmentacji w celu sprawdzenia spójności regionów odpowiadającym poszczególnym obiektom. Następnie dla każdego regionu określa się prymityw np. prostopadłościan, który następnie jest zoptymalizowany pod kątem odpowiadania danym głębi i maski.
Ta metoda jest bardziej efektywna niż algorytmy iteracyjne dla prostych obiektów, o nieskomplikowanym kształcie. Zależy ona jednak od jakości maski semantycznej i chmury punktów.

#### 3.1.4 Metoda wyrównywania wbudowana w bibliotekę realsense



### 3.2 Sposób wizualizacji danych

Jednym z najczęściej spotykanych sposobów wizualizacji danych zawierających informacje o głębi scenerii jest stosowanie chmury punktów. Polega ona na umiejscowieniu punktów np. w trójwymiarowym układzie odniesienia w celu przekazania informacji o głębi. W celu dodania innych informacji często dane punkty są kolorowane na różne barwy. Dzięki temu można łatwo zwizualizować zdjęcia wraz z  zarejestrowaną głębią. Kolory mogą również być przypisane do danej etytkiety przez co otrzymuje się zsegmentowaną chmurę pubnktów.

## 4. Program tworzący i operujący na trójwymiarowej mapie semantycznej

### 4.1 Opis programu

Po uzyskaniu obrazu z kamery należy podać go operacji wytrenowanym modelem w celu segmentacji semantycznej. W efekcie otrzymano informacje na temat przynależności danego piksela obrazu do klasy np. człowieka, samochodu itp. Na tej podstawie tworzona jest maska segmentacji. Kolejnym etapem jest dodanie informacji o głębi z czujników na segmentowany obraz.

Program został napisany w języku Python. Wykorzystano następujące biblioteki:

- pyrealsense2 - biblioteka pozwalająca na łatwy dostęp do obrazu i informacji o jego głębi z kamery Intel Realsense
- NumPy - biblioteka do obliczeń numerycznych
- OpenCV - otwartoźródłowa biblioteka do operacji na obrazach
- PyTorch - biblioteka udostępniająca pretrenowane modele AI z frameworka transformers
- TorchVision - biblioteka rodziny PyTorch udostępniająca gotowe datasety, wytrenowane modele wyspecjalizowane w używaniu przy wizji komputerowej.
- Ultralitics - biblioteka udostępniająca gotowe modele YOLO wyspecjalizowane w używaniu przy wizji komputerowej.
- Matplotib - biblioteka szeroko stosowana do rysowania wykresów i wizualizacji danych.
- Open3d - biblioteka służąca do operowania na chmury punktów.

Po uruchomieniu program inicjuje wybrany wytrenowany model oraz przygotowuje niezbędną konfigurację do obsługi kamery Realsense.  Po sprawdzeniu obecności sprzętu uruchamiana jest pętla, której zadaniem jest zniwelowanie degradacji kolorów, która wystpuje od razu po włączeniu kamery i zanika z czasem. Po odczekaniu nieznacznej chwili pobierane jest zdjęcie i informacje o głębi, które posłużą do stworzenia mapy.

Operacje na obrazie:

1. Transformacja obrazu na tablicę bibliotekii Numpy
2. Generowanie kolorów, którymi będą kolorowane dane piksela w celu oznaczenia przynależności do klasy
3. Segmentacja obrazu z naniesieniem kolorów
4. Wyświetlanie obrazu pobranego oraz po segmentacji
5. Naniesienie chmury punktów 3d z czujników odległości
6. Zapis otrzymanych punktów do pliku o rozszerzeniu .ply

### 4.2 Użyte modele

- resnet50
- resnet101
- mobilenet_v3_large
- vgg16
- modele rodziny yolov8-seg, yolo9-seg

### 4.3 Wizualizajc adanych

## Przypisy

 1. Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9404-9413)

 2. Hu, J., Huang, L., Ren, T., Zhang, S., Ji, R., & Cao, L. (2023). You Only Segment Once: Towards Real-Time Panoptic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 17819-17829)

 3. [Strona prezentująca kemerę](https://www.intelrealsense.com/depth-camera-d435/)

 4. [Github z 3d semantic map](https://github.com/shichaoy/semantic_3d_mapping?tab=readme-ov-file) - powiązana prac: Shichao Yang, Yulan Huang and Sebastian Scherer (2017) Semantic 3D Occupancy Mapping through Efficient High Order CRFs

 5. [SOTA z bechmarkami systemów panoptycznych](https://paperswithcode.com/task/panoptic-segmentation)

 6. [SOTA z benchmarkami systemów semantycznych](https://paperswithcode.com/task/semantic-segmentation)

 7. [SOTA z benchmarkami systemów semnatycznych 3D](https://paperswithcode.com/task/3d-semantic-segmentation)

 8. [Transformers by huggingface](https://github.com/huggingface/transformers)  [papier](https://aclanthology.org/2020.emnlp-demos.6.pdf)

 9. [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file)

 10. [Datacron2](https://github.com/facebookresearch/detectron2) [papier](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

 11. [mmdetection](https://github.com/open-mmlab/mmdetection)
 
 12. [OneFormer3d](https://arxiv.org/pdf/2311.14405v1), [wpis w paperwithcode.com](https://paperswithcode.com/paper/oneformer3d-one-transformer-for-unified-point) 

## Notatki z przeglądu literatury

### Segmentacje - definicje

- instance segmentation - detect and segment each object instance
- semantic segmentation - assign a clas label to each pixel
- panoptic segmentation - task that involves assigning a semanic label and an identity to each pixel of an input image
- semantic label - labels that classifie stuff like sky, road and things like person, cars.

### Semantic 3D Occupancy Mapping through Efficient High Order CRFs

Gdzie stosuje się 3D mapy semantyczne - w wielu zadaniach związanych z robotyką np. autonomiczne poruszanie się. Jest to zadania trudne, bo wymaga jednoczesnego wykonywania obliczeń gemoetrycznych jaki i segmentacji semantycznej obrazu. Szybkość wykonywania segmentacji semantycznej na dwuwymiarowych obrazach znacząco poprawiła się wraz z rozwojem konwolucyjnych sieci neuronoych, lecz nadal napotyka się problemy z dokładnością rozróżniania okluzji i cieni.
W tej pracy posłużono się dwoma obrazami RGB, a nie kamerą RGBD. Uargumentowano wybór tym, owe urządzenia można stosować jedynie w zamkniętych pomieszczeniach, na małym obszarze roboczym. Użyto konwolucyjnej sieci neuronowej w celu przetransferowania etykiet semantycznych z obrazu 2D na przestrzeń 3D. W efekcie zrekonstruowano mapę 3D na podstaiwe dwóch obrazów 2D i dokonano segmentacji jej obiektów.

### Panoptic Segmentation

Segmentacja panoptyczna - przypisanie etykiety do każdrgo pixela obrazu
Propozycja miary jakości panoptycznej - ujęcie wydajności każdej z etykiet w ujednolicony i interpretowalny sposób. Miara jakości predykcji panoptycznej w stosunku do "ground truth". IoU (intersection over union) >0.5,następnie suma ważona FP, FN, FP ze wszytskich etykiet.
Instance segmentation - wykrywanie obieków - zakreślanie je w bounding boxy
Segmentqacja panoptyczna - segmentacja semantyczna wraz z segmentacją instancji - każdemu pixelowi jest przypisywana klasa jak i instacncja np. każde auto ma label auto, ale są rozpoznawane poszczegolne auta jako auto A, B, C itd. Pixele o tej samej etykiecie jak i ID instacji należą do tego samego objektu.
Użyte datasety: PASCAL VOD, COCO, Cityscapes, ADE20k, Mapillary Vistas

Zbiór etykiet things i etykiet stuff się wyklucza, ale razem tworzą zbór etykiet semsntycznych.

### Yout Only Segment Once: Toward Reals-Time Panoptic Segmentation

YOSO - real time segmentation framework. Predykcja mask poprzez konwolucję i mapę cech obrazu (feature image map). Chwalą się, że należy tylko raz dokonać segmentacji semantycznej i instancji.
Opisują Real-Time Instance Segmentation, Real-Time Panoptic Segmentation, Real-Time Semantic Segmentation. Przytacają przykładu powiązanych prac o tej tematyce.
Opisują podstaę działania YOSO - piramidę agregującą cechy (feature pyramid aggregator), która kompresuje i agreguje wielowarstwowe mapy cech do jednowarstwowej.
Użyte datasety: COCO, Cityscapes, ADEK20K, Mapilliary Vistas.
Metryka ewaluacji: Panoptic quality, która może zostać zdekomponowana do segmentation quality i recognition quality.
Resultat: udao się osiągnąć lepsze resultaty niż w innych frameworkach przy zachowaniu "competitive panoptic quality preformence"

### PanoOcc: Unified Occupancy Representation for Camera-based 3D Panoptic Segmentation

PanoOcc - metoda oparta na agregacji informacji z wokseli w celu zrozumienia

### Point Transformer V3: Simpler, Faster, Stronger

Zasada działania:
1. Chmura punktów jest normalizowana przed podaniem ich do operatora. Ujednolica to dane do treningu modelu. Stosowana jest normalizacja warstw zamiast normalizacja batchy.
2. Nadal używany jest grid pooling wzięty z poprzedniego modelu PTv2.
3. Dodatkowa warstwa konwolucyjna pozwalająca na zamianę kodowania pozycji (Relative Position Encoding) na xCPE (enhnced Conditional Position Embedding), co daje lepsze wyniki przy minimalnie większym nakładzie obliczeniowym

Model jest rozwinięciem idei stojącej za PTv2. Jest szybszy a zarazem zachowuje elastyczność i prostotę.

### OneFormer3D: One Transformer for Unified Point Cloud Segmentation

Użycie backbone z U-Neta i połączenie z dekoderem transformer w celu zwiększenia wydajności (czasu i pamięci) - pierwsze takie połączenie.
Framework jest dziedziczony z SPFormer z uwagi na prostotę pipelineu, mały nakład pamięci, szybko interface (fast interface).
Zasada działania:
1. Najpierw 3d U-Net wyodrębnia cechy z punktów. Analizowany jest kolor RGB punktu oraz jego położenie w przestrzeni.
2. Uśrednienie cech punktów przez "flexible pooling" i łączenie punktów w "superpunkty". 
3. Cechy superpunktów są podawane do dekodera transformera - dokładniej to 6 warstw dekodera. Oprócz nich można również podać zapytania semantyczne i instancyjne w celu trenowania modelu.
4. Decoder zwraca zestaw kerneli. Każdy z nich reprezentuje segmentowaną maskę obiektu/przedmiotu. Może to być maska po segmentacji ponoptycznej, semantycznej lub instancyjnej.

Na moment pisania tego fragmentu pracy OneFormer3D jest notowany jako najlepszy model, który został sprawdzony na datasecie S3DIS pod kątem segmentacji semantycznej 3D oraz jako najlepszy model pod kontem segmentacji panoptycznej na datasecie ScanNet.

