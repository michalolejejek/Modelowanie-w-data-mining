# Detektor Pojazdów (Vehicle Detector)

Program do automatycznego wykrywania pojazdów na zdjęciach przy użyciu modelu YOLOv5. Program wykrywa następujące typy pojazdów: samochody, ciężarówki, motocykle, autobusy i rowery.

## Wymagania systemowe

- Python 3.8 lub nowszy
- Git (do pobrania YOLOv5)
- Połączenie internetowe (do pierwszego pobrania modelu)

## Instalacja

1. Sklonuj to repozytorium:
```bash
git clone [twój-link-do-repo]
cd [nazwa-folderu]
```

2. Sklonuj repozytorium YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5
```

3. Zainstaluj wymagane biblioteki:
```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install pillow
pip install numpy


## Struktura projektu
```
.
├── car_vision.py
├── yolov5/
└── README.md


## Użycie

1. Przygotuj zdjęcie, na którym chcesz wykryć pojazdy

2. Uruchom program z linii komend:

python car_vision.py sciezka/do/twojego/obrazu.jpg


Na przykład:

python car_vision.py test_image.jpg


## Wyniki

Program:
- Wyświetli w konsoli liczbę wykrytych pojazdów z podziałem na typy
- Zapisze nowy obraz z zaznaczonymi pojazdami jako `detected_vehicles.jpg`
- Wyświetli okno z wizualizacją wykrytych pojazdów

## Obsługiwane typy pojazdów

- Samochody (car)
- Ciężarówki (truck)
- Motocykle (motorcycle)
- Autobusy (bus)
- Rowery (bicycle)

## Rozwiązywanie problemów

Jeśli napotkasz błąd "No module named 'torch'":
1. Upewnij się, że PyTorch jest zainstalowany:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```
