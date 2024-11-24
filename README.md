# Easy 3D Semantic Map

English version: [README_en.md](/README_en.md)

Niniejsze repozytorium jest powiązane z pracą dyplomową "Tworzenie i operowanie na trójwymiarowej mapie semantycznej".

Celem repozytorium jest uproszczenie procesu tworzenia map semantycznych przy użyciu kamery RealSense D435f.

## Środowisko

Repozytorium było uruchamiane na Pythonie 3.9.18. Z uwagi na kompatybilność bibliotek nie zaleca się używania wersji Pythona wyższej niż 3.10.
W celu uruchomienia należy sklonować repozytorium.
Zaleca się zbudowanie nowego środowiska wirtualnego. Niezbędne biblioteki użyte w tym projekcie są zawarte w [requirements.txt](/requirements.txt). W celu instalacji należy użyć komendy
```
pip install -r requirements.txt
```

## Przykłady użycia

Przygotowano 3 programy prezentujące działanie biblioteki.

### [example_1.py](/example_1.py)

Przykład 1 prezentuje sposób obsługi kamery RealSense D435f. Pokazano w jaki sposób sprawdzić obecność kamery, uzyskać informacje o jej konfiguracji, pobrać obraz kolorowy i odczyt głębi scenerii jak i wyświetlić chmurę punktów.

### [example_2.py](/example_2.py)

Przykład 2 ukazuje sposób używania funkcji z modelami segmentacji obrazu i estymacji głebi na obrazie uzyskanym z kamery.

### [example_3.py](/example_3.py)

Przykład 3 pokazuje w jaki sposób tworzona jest trójwymiarowa mapa semantyczna z dowolnego wczytanego zdjęcia.

### [example_0.py](/example_0.py)

Dodatkowo dodano tzw. przykład 0, który jest skryptem operującym wyłacznie na funkcjach biblioteki `pyrealsesne2` w celu bieżącego podglądu obrazu kolorowego jak i głebi z kamery.


