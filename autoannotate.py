# gerar_anotacoes.py
from ultralytics.data.annotator import auto_annotate

# Parâmetros de configuração
caminho_imagens = "carsdataset/ambulance/"  # Pasta com as imagens a anotar
caminho_saida = "carsdataset/ambulance-labels"  # Pasta onde as anotações serão salvas
modelo_deteccao = "yolo11n.pt"  # Modelo YOLO para detecção
modelo_sam = "mobile_sam.pt"    # Modelo SAM para segmentação
dispositivo = "cuda"             # Use "cuda" para GPU ou "cpu" para CPU

auto_annotate(
    data=caminho_imagens,
    det_model=modelo_deteccao,
    sam_model=modelo_sam,
    device=dispositivo,
    output_dir=caminho_saida
)
