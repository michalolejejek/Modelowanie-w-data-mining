import sys
sys.path.append('./yolov5')  
import torch
import cv2
import numpy as np
import argparse
from PIL import Image


class VehicleDetector:
    def __init__(self):
        #model Yolov5
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.vehicle_classes = ['car', 'truck', 'motorcycle', 'bus', 'bicycle']
        
    def detect_vehicles(self, image_path):
        # Wczytanie obrazu
        image = Image.open(image_path)
        
        # Wykonanie detekcji
        results = self.model(image)
        
        # Filtrowanie wyników tylko dla pojazdów
        vehicles = []
        for detection in results.xyxy[0]:  # results.xyxy[0] zawiera współrzędne bbox, pewność i klasę
            x1, y1, x2, y2, conf, cls = detection
            class_name = self.model.names[int(cls)]
            
            if class_name in self.vehicle_classes and conf > 0.5:
                vehicles.append({
                    'type': class_name,
                    'confidence': conf.item(),
                    'bbox': (x1.item(), y1.item(), x2.item(), y2.item())
                })
        
        return vehicles

    def draw_detections(self, image_path, vehicles):
        # Wczytanie obrazu do rysowania
        image = cv2.imread(image_path)
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            label = f"{vehicle['type']} {vehicle['confidence']:.2f}"
            
            # Rysowanie bbox
            cv2.rectangle(image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Dodanie etykiety
            cv2.putText(image, label, 
                       (int(x1), int(y1-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        return image

def main():
    # Dodanie parsera argumentów
    parser = argparse.ArgumentParser(description='Wykrywanie pojazdów na zdjęciu')
    parser.add_argument('image_path', type=str, help='Ścieżka do zdjęcia')
    args = parser.parse_args()

    # Sprawdzenie czy plik istnieje
    try:
        detector = VehicleDetector()
        vehicles = detector.detect_vehicles(args.image_path)
        
        # Zliczanie typów pojazdów
        vehicle_counts = {}
        for v in vehicles:
            vehicle_type = v['type']
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        
        print(f"\npojazdów jest: {len(vehicles)} :")
        for vehicle_type, count in vehicle_counts.items():
            print(f"- {vehicle_type}: {count}")
        
        #generowanie obrazu z detekcjami
        result_image = detector.draw_detections(args.image_path, vehicles)
        output_path = 'detected_vehicles.jpg'
        cv2.imwrite(output_path, result_image)
        print(f"\nZapisano obraz z detekcjami do: {output_path}")
        cv2.imshow('Wykryte pojazdy', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {args.image_path}")
    except Exception as e:
        print(f"Wystąpił błąd: {str(e)}")

if __name__ == '__main__':
    main()
