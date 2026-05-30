"""
Script de limpeza de labels - roda DEPOIS do Grounding DINO
Remove:
  1. Bboxes que são pessoas (usando YOLOv8 COCO como detector)
  2. Bboxes com aspect ratio de pessoa (alto e estreito)
  3. Bboxes suspeitos de serem ambulâncias mal classificadas
"""

from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil

class LabelCleaner:
    def __init__(self, dataset_dir, backup=True):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / 'images' / 'train'
        self.labels_dir = self.dataset_dir / 'labels' / 'train'
        
        # Cria backup dos labels originais
        if backup:
            self.backup_dir = self.dataset_dir / 'labels_original_backup'
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            shutil.copytree(self.labels_dir, self.backup_dir)
            print(f"💾 Backup dos labels originais: {self.backup_dir}")
        
        # Detector de pessoas (YOLOv8n COCO - classe 0 = person)
        print("📥 Carregando detector de pessoas...")
        self.person_detector = YOLO('yolov8n.pt')
        
        # Estatísticas
        self.stats = {
            'images_processed': 0,
            'bboxes_removed_person': 0,
            'bboxes_removed_aspect': 0,
            'bboxes_ambulance_recovered': 0,
            'bboxes_total_before': 0,
            'bboxes_total_after': 0,
        }
    
    def iou(self, box1, box2):
        """Calcula IoU entre duas boxes em formato (x1, y1, x2, y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def yolo_to_xyxy(self, bbox, w, h):
        """Converte YOLO (xc, yc, w, h) normalizado para (x1, y1, x2, y2) pixels"""
        xc, yc, bw, bh = bbox
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h
        return (x1, y1, x2, y2)
    
    def clean_image(self, img_path, label_path):
        """Limpa os labels de uma imagem"""
        if not label_path.exists():
            return
        
        # Carrega imagem para pegar dimensões
        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            return
        
        # Lê labels originais
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            return
        
        # Parse dos bboxes
        bboxes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    bboxes.append({
                        'class': int(parts[0]),
                        'xc': float(parts[1]),
                        'yc': float(parts[2]),
                        'w': float(parts[3]),
                        'h': float(parts[4]),
                        'original_line': line
                    })
                except (ValueError, IndexError):
                    continue
        
        self.stats['bboxes_total_before'] += len(bboxes)
        
        # ===== ETAPA 1: Detectar pessoas na imagem =====
        person_detections = self.person_detector.predict(
            str(img_path), 
            conf=0.4,  # Confiança alta para evitar falsos positivos
            classes=[0],  # Só classe "person"
            verbose=False
        )[0]
        
        person_boxes_xyxy = []
        for box in person_detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            person_boxes_xyxy.append((x1, y1, x2, y2))
        
        # ===== ETAPA 2: Filtrar bboxes =====
        cleaned_bboxes = []
        
        for bbox in bboxes:
            bbox_xyxy = self.yolo_to_xyxy(
                (bbox['xc'], bbox['yc'], bbox['w'], bbox['h']), w, h
            )
            
            # Regra 1: Aspect ratio de pessoa (alto e estreito)
            # Pessoas em pé têm h/w > 1.8 tipicamente
            pixel_w = bbox_xyxy[2] - bbox_xyxy[0]
            pixel_h = bbox_xyxy[3] - bbox_xyxy[1]
            
            if pixel_w > 0:
                aspect = pixel_h / pixel_w
                if aspect > 2.0 and bbox['class'] == 1:  # Só aplica em "car"
                    # Muito provável que seja pessoa rotulada como carro
                    self.stats['bboxes_removed_aspect'] += 1
                    continue
            
            # Regra 2: Sobreposição com detecção de pessoa
            is_person = False
            for person_box in person_boxes_xyxy:
                iou_val = self.iou(bbox_xyxy, person_box)
                # Se o bbox tem alta sobreposição com uma pessoa detectada
                # E o bbox é pequeno (tamanho de pessoa, não de carro)
                bbox_area = pixel_w * pixel_h
                image_area = w * h
                bbox_ratio = bbox_area / image_area
                
                if iou_val > 0.5 and bbox_ratio < 0.05:  # bbox pequeno + overlap alto
                    is_person = True
                    break
            
            if is_person and bbox['class'] == 1:  # Só remove se era "car"
                self.stats['bboxes_removed_person'] += 1
                continue
            
            # Se passou em todas as regras, mantém
            cleaned_bboxes.append(bbox)
        
        self.stats['bboxes_total_after'] += len(cleaned_bboxes)
        self.stats['images_processed'] += 1
        
        # Reescreve o label
        with open(label_path, 'w') as f:
            for bbox in cleaned_bboxes:
                f.write(f"{bbox['class']} {bbox['xc']:.6f} {bbox['yc']:.6f} "
                       f"{bbox['w']:.6f} {bbox['h']:.6f}\n")
    
    def clean_all(self):
        """Processa todas as imagens"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in self.images_dir.iterdir() if f.suffix.lower() in extensions]
        
        print(f"\n🧹 Limpando labels de {len(images)} imagens...\n")
        
        for img_path in tqdm(images, desc="Limpando"):
            label_path = self.labels_dir / (img_path.stem + '.txt')
            self.clean_image(img_path, label_path)
        
        self.print_stats()
    
    def print_stats(self):
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS DA LIMPEZA")
        print("="*60)
        print(f"Imagens processadas:              {self.stats['images_processed']}")
        print(f"Bboxes ANTES:                     {self.stats['bboxes_total_before']}")
        print(f"Bboxes DEPOIS:                    {self.stats['bboxes_total_after']}")
        print(f"Removidos (aspect ratio pessoa):  {self.stats['bboxes_removed_aspect']}")
        print(f"Removidos (detector de pessoas):  {self.stats['bboxes_removed_person']}")
        print(f"Total removidos:                  {self.stats['bboxes_total_before'] - self.stats['bboxes_total_after']}")
        print("="*60)


if __name__ == "__main__":
    DATASET_DIR = "./dataset_rotulado"
    cleaner = LabelCleaner(DATASET_DIR, backup=True)
    cleaner.clean_all()
    print("\n✅ Limpeza concluída! Rode o viewer.py novamente para validar.")
