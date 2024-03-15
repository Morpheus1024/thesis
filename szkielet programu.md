# Ideowy szkielet programu

## Etapy tworzenia semantycznej mapy 3d otoczenia

### 1. Pozyskanie danych
- zdjęcia z kamery
- mapa głębi

### 2. Preprocessing
- kalibracja - komensacja błędu geometrycznego rozstawienia czujników
- segmentacja obrazu
- filtracja mapy głębi

### 3. Rekonstrukcja 3D
- punkty chury z mapy głębi
- siatka 3D - na podstawie chmury punktów tworzy się siatkę reprezentującą powierzchnie

### 4. Semantyczna segmentacja
- klasyfikacja każdego z segmentów
- przypisanie etykiet do obiektów - zastosować wiele modeli

### 5. Integracja
- połączenie danych z segmentacji, klasyfikacji i siatki 3D w celu stworzenia mapy semantycznej
- wizializacja danch