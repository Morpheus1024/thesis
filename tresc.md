# Praca inżynierska

## 1. Wstęp

### 1.1 Cel Pracy

W niniejszej pracy przedstawiono proces tworzenia i operowania na trójwymiarowej mapie semantycznej na podstawie danych pochodzących z kamery RGP-D Intel RealSense.
Używając dostępnych bibliotek udostępnionych przez producenta odczytano dane z kamery, które posłużyły jako podstawa do tworzenia mapy semantycznej. Następnie, przez odpowiednią obróbkę danych jak i wykorzystanie dostępnych otwartoźródłowo modeli dokonano segmentacji obiektów widzianych przez kamerę.

### 1.2  Definicja segmentacji semantycznej

Segmentacja semantyczna jest podzadaniem segmentacji panoptycznej, którą definiuje się jako przypisanie każdemu pixelowi analizowanego obrazu etykiety semantycznej oraz identyfikacji każdej z instancji występującej na obrazie. Etykiety są zazwyczaj dzielone na te opisujące na obiektach policzalnych - ang. things - np. osoby, samochody, drzewa, oraz obiektach niepoliczalnych i amorficznych - ang. stuff - takie jak niebo, droga. [przypis 2. rozdział 1]. Operacje segmentacji wykonywane tych drugich są określane mianem segmentacji semantycznej [przypis 1. rozdział 1.].

### 1.3 Opis kamery stereosokopowej

Kamera wykorzystana w niniejszej pracy to Intel® RealSense™ Depth Camera D435. Wyposarzona jest ona w klasyczy obiektyw RGB jak i oprzyrządowanie do odczytania informacji o głębi obrazu. Wykorzystuje do tego rzutnik punktów widocznych w podczerwieni, których pozycja jest określana przez stereoskopowe czujniki podczerwieni. Producent określa odległość roboczą przyrządu od 30 cm do 3 m. [przypis 3. specyfikacja techniczna]

## 2. Przegląd używanych systemów

Z roku na rok powstają coraz nowsze sposoby i podejścia do segmentacji zdjęć i filmów. Poddawane są one testom wydajnościowym na otwartoźródłowych, powszechnie uznanych wśród społeczności zbiorach danych. Trudno nadążyć za najnowyszymi i najwydajniejszymi systemami z uwagi na szybkość zmian jakie zachodzą w tej dziedzinie.
W tym rozdzialę zostaną opisane jedne z najpopularniejszych używanych systemy segmentacji, które są aktywnie używane w celach segmentacji panoptycznej.

Do najpopularniejszych frameworków segmentacji panoptyczej można zaliczyć transformers [przypis 8], Pythorch Image Models [przypis 9], Datacron2 [przypis 10] oraz mmdetection [przypis 11].Wybrano je z uwagi na wsparcie społeczności, dostępność i popularność na repozytorium Github oraz ciągły rozwój w celu osiągnięcia coraz lepszych wyników wydajności.

### Transformers

Transformers oddaje w ręce użytkownika API, które pozwalają na używanie już wytrenowanych modelu. Framework jest szeroko stosowany w dziedzianch związanych z NLP, audio czy chociażby z wizją komputerową.

Transformers został zaprojektowany by jak najlepiej odwzorować modele potokowe tzn. dokonać wstępnej obróbki danych, poddać je działaniu modelu i dokonać predykcji. Każdy z modeli został zdefiniowan poprzez trzy bloki stanowące rdzeń działąnia całego rameworka: blok tokenizajic danych, blok transformacji (od którego wzięła się nazwa frameworka), oraz z bloku głowy/głów.

Blok tokenizacji - również nazywany tokenizerem - jest odpowiedzialny za nadawanie stokenizowanych klas, które są niezbędne do pracy każdgo modelu. Klasy mogą być już predefiuniowane, ale mogą również zostać dodane przez użytkownika. Przechowuje on listę mapującą token do indeksu.

Blok transformacji ma za zadanie wykonywać zadanie modelu np. generowanie, rozumowanie na podstawie klas powstałych w wyniku tokenizacji w poprzednim bloku. Architektury modeli zostały w taki sposób dobrane, by była możliwość łatwego podnieniania ich bloku transformacji.

Blok głów jest odpowiedzialny za dostosowanie danych otrzymanych z bloku transformacji do danych wyjściowych dostosowanych do danego zadania np. współrzędne bounding boxa. Dodaje on do klasy bazowej warstwę wyjścia oraz funkcję straty. Niektóre bloki obsługują również dodatkowe funkcje takie jak próbkowanie w celu wykonania powieżonego im zadania.

Celem autorów Transformers było stowrzenie hubu wytrenowanych modeli w celu ułatwienia dostępu do nich oraz łatwego aplikowania ich do projektów użytkowników. W 2020 roku hub oferował ponad 2000 modeli, w tym BERT i GPT-2. Na czas pisania tej pracy modeli jest ponad 660000.

Modele dostępne w Transformers można zainstalować poprzez instalację biblioteki PyTorch, Tensorflow oraz Flex jak i poprzez bezpośrednie pobranie ze strony projektu na Githubie.

### Pythorch Image Models

Pythorch Image Models jest kolejnym otwartoźródłowym frameworkem, który daje dostęp do wytrenowanych już modeli skupionych wokół wizji komputerowej.

## Prypisy

 1. Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9404-9413)

 2. Hu, J., Huang, L., Ren, T., Zhang, S., Ji, R., & Cao, L. (2023). You Only Segment Once: Towards Real-Time Panoptic Segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 17819-17829)

 3. [Strona prezentujaca kemerę](https://www.intelrealsense.com/depth-camera-d435/)

 4. [Github z 3d semantic map](https://github.com/shichaoy/semantic_3d_mapping?tab=readme-ov-file) - powiązana prac: Shichao Yang, Yulan Huang and Sebastian Scherer (2017) Semantic 3D Occupancy Mapping through Efficient High Order CRFs

 5. [SOTA z bechmarkami systemów panoptycznych](https://paperswithcode.com/task/panoptic-segmentation)

 6. [SOTA z benchmarkami systemów semantycznych](https://paperswithcode.com/task/semantic-segmentation)

 7. [SOTA z benchmarkami systemów semnatycznych 3D](https://paperswithcode.com/task/3d-semantic-segmentation)

 8. [Transformers by huggingface](https://github.com/huggingface/transformers)  [papier](https://aclanthology.org/2020.emnlp-demos.6.pdf)

 9. [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file)

 10. [Datacron2](https://github.com/facebookresearch/detectron2) [papier](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

 11. [mmdetection](https://github.com/open-mmlab/mmdetection)

## Notatki z tekstów skazanych przez Doktora

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
