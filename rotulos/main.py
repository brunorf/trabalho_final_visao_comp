import torch
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import shutil
import inspect

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINOLabeler:
    def __init__(self, output_dir='./dataset_rotulado', device='cuda'):
        self.output_dir = Path(output_dir)
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'visual_samples').mkdir(parents=True, exist_ok=True)
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Device: {self.device}")
        
        print("📥 Carregando Grounding DINO...")
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        print("✅ Modelo carregado!")
        
        # ⭐ PROMPT COM DUAS CLASSES
        self.text_prompt = "ambulance. car. vehicle. truck. van"
        
        # Mapeamento de classes
        self.class_mapping = {
            'ambulance': 0,
            'car': 1,
            'vehicle': 1,  # vehicle genérico = car
            'truck': 1,
            'van': 1
        }
        
        self.class_names = ['ambulance', 'car']
        
        # Thresholds
        self.box_threshold = 0.25
        self.text_threshold = 0.25
        
        # Detecta API
        self._detect_api()
        
        self.stats = {
            'processed': 0, 
            'with_detections': 0, 
            'without': 0, 
            'ambulance_boxes': 0,
            'car_boxes': 0
        }
    
    def _detect_api(self):
        """Detecta parâmetros disponíveis na API"""
        sig = inspect.signature(self.processor.post_process_grounded_object_detection)
        self.api_params = list(sig.parameters.keys())
        print(f"📋 Parâmetros: {self.api_params}")
        
        if 'box_threshold' in self.api_params:
            self.box_param_name = 'box_threshold'
        elif 'boxes_threshold' in self.api_params:
            self.box_param_name = 'boxes_threshold'
        elif 'threshold' in self.api_params:
            self.box_param_name = 'threshold'
        else:
            self.box_param_name = None
    
    def classify_label(self, label_text):
        """Classifica o texto da label em uma das 2 classes"""
        label_lower = label_text.lower()
        
        # Verifica se contém "ambulance"
        if 'ambulance' in label_lower or 'emergency' in label_lower:
            return 0  # ambulance
        
        # Caso contrário, é carro/veículo
        return 1  # car
    
    def process_image(self, img_path):
        """Processa uma imagem e detecta AMBULÂNCIAS E CARROS"""
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Erro ao abrir {img_path}: {e}")
            return None
        
        inputs = self.processor(
            images=image,
            text=self.text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pós-processamento
        kwargs = {
            'outputs': outputs,
            'input_ids': inputs.input_ids,
            'target_sizes': [image.size[::-1]],
        }
        
        if self.box_param_name:
            kwargs[self.box_param_name] = self.box_threshold
        if 'text_threshold' in self.api_params:
            kwargs['text_threshold'] = self.text_threshold
        
        try:
            results = self.processor.post_process_grounded_object_detection(**kwargs)[0]
        except TypeError:
            # Fallback
            kwargs_fallback = {
                'outputs': outputs,
                'input_ids': inputs.input_ids,
                'target_sizes': [image.size[::-1]],
            }
            results = self.processor.post_process_grounded_object_detection(**kwargs_fallback)[0]
        
        # Processa resultados
        bboxes = []
        w, h = image.size
        
        scores = results.get("scores", [])
        labels = results.get("labels", [])
        boxes = results["boxes"]
        
        for i, box in enumerate(boxes):
            score = float(scores[i]) if len(scores) > i else 1.0
            label_text = labels[i] if len(labels) > i else "vehicle"
            
            # Filtragem manual
            if score < self.box_threshold:
                continue
            
            x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
            
            # Filtra bboxes muito pequenas
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h
            if bbox_w < 0.02 or bbox_h < 0.02:
                continue
            
            # ⭐ CLASSIFICA EM AMBULÂNCIA OU CARRO
            class_id = self.classify_label(label_text)
            
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            
            bboxes.append({
                'class': class_id,
                'class_name': self.class_names[class_id],
                'x_center': x_center,
                'y_center': y_center,
                'width': bbox_w,
                'height': bbox_h,
                'score': score,
                'label_text': label_text,
                'pixel_bbox': (x1, y1, x2, y2)
            })
        
        return {'image': image, 'bboxes': bboxes}
    
    def save_yolo_format(self, img_path, bboxes, img):
        """Salva no formato YOLO com 2 classes"""
        dst_img = self.output_dir / 'images' / 'train' / img_path.name
        img.save(dst_img)
        
        label_path = self.output_dir / 'labels' / 'train' / (img_path.stem + '.txt')
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                # Formato: class_id x_center y_center width height
                f.write(f"{bbox['class']} {bbox['x_center']:.6f} "
                       f"{bbox['y_center']:.6f} {bbox['width']:.6f} "
                       f"{bbox['height']:.6f}\n")
    
    def save_visual_sample(self, img_path, bboxes, img, max_samples=30):
        """Salva amostra visual com cores diferentes para cada classe"""
        if len(list((self.output_dir / 'visual_samples').glob('*'))) >= max_samples:
            return
        
        draw = ImageDraw.Draw(img)
        
        colors = {
            0: 'lime',      # Verde para ambulância
            1: 'cyan'       # Ciano para carro
        }
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['pixel_bbox']
            color = colors[bbox['class']]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1-15), 
                     f"{bbox['class_name']} {bbox['score']:.2f}", 
                     fill=color)
        
        img.save(self.output_dir / 'visual_samples' / img_path.name)
    
    def process_folder(self, input_folder, folder_name=""):
        """Processa uma pasta de imagens"""
        input_path = Path(input_folder)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        print(f"\n📂 Processando {folder_name}: {input_folder}")
        print(f"📷 {len(images)} imagens")
        
        for img_path in tqdm(images, desc=f"Rotulando {folder_name}"):
            result = self.process_image(img_path)
            if result is None:
                continue
            
            self.stats['processed'] += 1
            
            if result['bboxes']:
                self.stats['with_detections'] += 1
                
                # Conta detecções por classe
                for bbox in result['bboxes']:
                    if bbox['class'] == 0:
                        self.stats['ambulance_boxes'] += 1
                    else:
                        self.stats['car_boxes'] += 1
                
                self.save_yolo_format(img_path, result['bboxes'], result['image'])
                self.save_visual_sample(img_path, result['bboxes'], result['image'].copy())
            else:
                self.stats['without'] += 1
        
        print(f"✅ {folder_name} concluído")
    
    def print_stats(self):
        print("\n" + "="*60)
        print("📊 ESTATÍSTICAS FINAIS")
        print("="*60)
        print(f"Imagens processadas:        {self.stats['processed']}")
        print(f"Com detecções:              {self.stats['with_detections']}")
        print(f"Sem detecções:              {self.stats['without']}")
        print(f"Total bboxes ambulância:    {self.stats['ambulance_boxes']}")
        print(f"Total bboxes carro:         {self.stats['car_boxes']}")
        print(f"Total bboxes:               {self.stats['ambulance_boxes'] + self.stats['car_boxes']}")
        print("="*60)
    
    def create_yaml_config(self):
        """Cria YAML com 2 classes"""
        yaml_content = f"""# Dataset com 2 classes: ambulância e carro
path: {self.output_dir.absolute()}
train: images/train
val: images/train  # Use pasta separada se tiver

nc: 2
names:
  0: ambulance
  1: car
"""
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"✅ YAML criado: {yaml_path}")


if __name__ == "__main__":
    # ⚠️ AJUSTE ESSES CAMINHOS
    AMBULANCE_FOLDER = "/home/bruno.rogerio/.cache/kagglehub/datasets/mahmoudshaheen1134/ambulance-dataset/versions/1/carsdataset/ambulance"
    CAR_FOLDER = "/home/bruno.rogerio/.cache/kagglehub/datasets/mahmoudshaheen1134/ambulance-dataset/versions/1/carsdataset/noambulance"  # Ajuste o caminho
    OUTPUT_DIR = "./dataset_rotulado"
    
    labeler = GroundingDINOLabeler(output_dir=OUTPUT_DIR)
    
    # Processa AMBAS as pastas
    labeler.process_folder(AMBULANCE_FOLDER, folder_name="ambulancias")
    labeler.process_folder(CAR_FOLDER, folder_name="carros")
    
    labeler.print_stats()
    labeler.create_yaml_config()
    
    print(f"\n✅ Dataset pronto!")
    print(f"📁 Valide visualmente em: {OUTPUT_DIR}/visual_samples/")
    print(f"   - Verde = ambulância")
    print(f"   - Ciano = carro")
