"""
Visualizador Web de Dataset Rotulado (YOLO format)
- Infinite scroll
- Thumbnails redimensionadas para carregamento rápido
- Clique para ampliar (mantendo bbox)
- Funciona em máquina remota via browser

Uso:
    python viewer.py --dataset ./dataset_rotulado --port 5000

Acesse: http://IP_DA_MAQUINA:5000
"""

import argparse
import json
import os
from pathlib import Path
from io import BytesIO
from functools import lru_cache

from flask import Flask, jsonify, request, send_file, render_template_string
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ========== CONFIGURAÇÃO GLOBAL ==========
DATASET_DIR = None
IMAGES_DIR = None
LABELS_DIR = None
THUMBS_DIR = None
CLASS_NAMES = ['ambulance', 'car']  # Ajuste se necessário
THUMB_SIZE = (480, 480)  # Tamanho máximo do thumbnail
BATCH_SIZE = 24  # Imagens por requisição

# Cores por classe (RGB)
CLASS_COLORS = {
    0: (50, 255, 50),    # Verde - ambulância
    1: (50, 200, 255),   # Ciano - carro
    2: (255, 100, 100),  # Vermelho - extra
    3: (255, 200, 50),   # Amarelo - extra
}

# ========== CACHE DE METADADOS ==========
_image_list_cache = None

def get_image_list():
    """Retorna lista de imagens com metadados (cacheado)"""
    global _image_list_cache
    if _image_list_cache is not None:
        return _image_list_cache
    
    print("📂 Escaneando dataset...")
    items = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue
        
        label_path = LABELS_DIR / (img_path.stem + '.txt')
        
        # Lê anotações
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            annotations.append({
                                'class': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
                        except (ValueError, IndexError):
                            continue
        
        items.append({
            'filename': img_path.name,
            'stem': img_path.stem,
            'n_boxes': len(annotations),
            'classes': list(set(a['class'] for a in annotations)),
        })
    
    # Ordena: primeiro as que têm mais bboxes
    items.sort(key=lambda x: (-x['n_boxes'], x['filename']))
    
    _image_list_cache = items
    print(f"✅ {len(items)} imagens encontradas")
    return items


# ========== PROCESSAMENTO DE IMAGENS ==========

def draw_bboxes_on_image(img, annotations, font=None):
    """Desenha bboxes na imagem PIL"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for ann in annotations:
        cls_id = ann['class']
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Converte de YOLO normalizado para pixels
        x1 = int((ann['x_center'] - ann['width'] / 2) * w)
        y1 = int((ann['y_center'] - ann['height'] / 2) * h)
        x2 = int((ann['x_center'] + ann['width'] / 2) * w)
        y2 = int((ann['y_center'] + ann['height'] / 2) * h)
        
        # Desenha retângulo
        line_width = max(2, min(w, h) // 200)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Label
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        text = f" {label} "
        
        # Tamanho do texto adaptativo
        try:
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=font)
        
        # Fundo do texto
        draw.rectangle(
            [x1, y1 - text_h - 4, x1 + text_w + 4, y1],
            fill=color
        )
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(0, 0, 0), font=font)


def load_annotations(stem):
    """Carrega anotações de uma imagem"""
    label_path = LABELS_DIR / f"{stem}.txt"
    annotations = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        annotations.append({
                            'class': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
                    except (ValueError, IndexError):
                        continue
    return annotations


def get_font(size):
    """Tenta carregar uma fonte TrueType, senão usa a padrão"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# ========== ROTAS DA API ==========

@app.route('/')
def index():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE, batch_size=BATCH_SIZE)


@app.route('/api/images')
def api_images():
    """Retorna lista paginada de imagens"""
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', BATCH_SIZE))
    filter_class = request.args.get('class', None)  # Filtrar por classe
    
    items = get_image_list()
    
    # Filtro opcional por classe
    if filter_class is not None:
        try:
            cls_id = int(filter_class)
            items = [it for it in items if cls_id in it['classes']]
        except ValueError:
            pass
    
    page = items[offset:offset + limit]
    
    return jsonify({
        'items': page,
        'offset': offset,
        'limit': limit,
        'total': len(items),
        'has_more': offset + limit < len(items),
    })


@app.route('/api/stats')
def api_stats():
    """Estatísticas do dataset"""
    items = get_image_list()
    class_counts = {}
    total_boxes = 0
    images_with_boxes = 0
    
    for it in items:
        if it['n_boxes'] > 0:
            images_with_boxes += 1
        for cls in it['classes']:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    return jsonify({
        'total_images': len(items),
        'images_with_boxes': images_with_boxes,
        'images_per_class': {
            CLASS_NAMES.get(k, f'class_{k}'): v 
            for k, v in class_counts.items()
        },
        'class_names': CLASS_NAMES,
    })


@app.route('/thumb/<filename>')
def serve_thumb(filename):
    """Serve thumbnail com bbox desenhado"""
    thumb_path = THUMBS_DIR / filename
    img_path = IMAGES_DIR / filename
    
    if not img_path.exists():
        return "Imagem não encontrada", 404
    
    # Gera thumbnail se não existir
    if not thumb_path.exists():
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
            
            # Desenha bboxes
            stem = Path(filename).stem
            annotations = load_annotations(stem)
            font_size = max(12, min(img.size) // 30)
            font = get_font(font_size)
            draw_bboxes_on_image(img, annotations, font=font)
            
            img.save(thumb_path, 'JPEG', quality=85, optimize=True)
        except Exception as e:
            print(f"❌ Erro ao gerar thumb de {filename}: {e}")
            return "Erro ao processar imagem", 500
    
    return send_file(thumb_path, mimetype='image/jpeg')


@app.route('/full/<filename>')
def serve_full(filename):
    """Serve imagem em resolução original com bbox"""
    img_path = IMAGES_DIR / filename
    
    if not img_path.exists():
        return "Imagem não encontrada", 404
    
    try:
        img = Image.open(img_path).convert('RGB')
        
        stem = Path(filename).stem
        annotations = load_annotations(stem)
        font_size = max(16, min(img.size) // 50)
        font = get_font(font_size)
        draw_bboxes_on_image(img, annotations, font=font)
        
        # Serve sem salvar em disco (pode ser grande)
        buf = BytesIO()
        img.save(buf, 'JPEG', quality=92)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        print(f"❌ Erro ao processar {filename}: {e}")
        return "Erro ao processar imagem", 500


# ========== HTML/JS FRONTEND ==========

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dataset Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a1a;
    color: #eee;
    padding: 20px;
  }
  header {
    position: sticky;
    top: 0;
    background: #1a1a1a;
    padding: 15px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #333;
    z-index: 10;
  }
  h1 { font-size: 22px; margin-bottom: 10px; }
  .stats {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    font-size: 14px;
    color: #aaa;
  }
  .stats span { background: #2a2a2a; padding: 5px 12px; border-radius: 4px; }
  .stats strong { color: #fff; }
  .filters {
    margin-top: 10px;
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .filters button {
    background: #2a2a2a;
    color: #eee;
    border: 1px solid #444;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }
  .filters button:hover { background: #3a3a3a; }
  .filters button.active { background: #4a90e2; border-color: #4a90e2; }
  
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 15px;
  }
  .card {
    background: #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  }
  .card img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
    background: #000;
  }
  .card-info {
    padding: 10px;
    font-size: 12px;
  }
  .card-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #ccc;
    margin-bottom: 4px;
  }
  .card-badges { display: flex; gap: 4px; flex-wrap: wrap; }
  .badge {
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-0 { background: #32ff32; color: #000; }
  .badge-1 { background: #32c8ff; color: #000; }
  .badge-2 { background: #ff6464; color: #fff; }
  
  #loader {
    text-align: center;
    padding: 30px;
    color: #888;
    font-size: 14px;
  }
  #sentinel { height: 1px; }
  
  /* Modal */
  .modal {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.95);
    z-index: 100;
    justify-content: center;
    align-items: center;
    padding: 20px;
    cursor: zoom-out;
  }
  .modal.active { display: flex; }
  .modal img {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 4px;
  }
  .modal-info {
    position: fixed;
    top: 20px;
    left: 20px;
    background: rgba(0,0,0,0.7);
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 13px;
    max-width: 60%;
    word-break: break-all;
  }
  .modal-close {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(255,255,255,0.15);
    color: #fff;
    border: none;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-size: 24px;
    cursor: pointer;
  }
</style>
</head>
<body>
<header>
  <h1>🖼️ Dataset Viewer</h1>
  <div class="stats" id="stats">Carregando...</div>
  <div class="filters" id="filters"></div>
</header>

<div class="grid" id="grid"></div>
<div id="loader">Carregando...</div>
<div id="sentinel"></div>

<div class="modal" id="modal">
  <button class="modal-close" onclick="closeModal()">×</button>
  <div class="modal-info" id="modal-info"></div>
  <img id="modal-img" src="" alt="">
</div>

<script>
const BATCH_SIZE = {{ batch_size }};
let offset = 0;
let loading = false;
let hasMore = true;
let currentFilter = null;
let classNames = [];

async function loadStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  classNames = data.class_names;
  
  let html = `<span>Total: <strong>${data.total_images}</strong></span>`;
  html += `<span>Com anotações: <strong>${data.images_with_boxes}</strong></span>`;
  for (const [cls, count] of Object.entries(data.images_per_class)) {
    html += `<span>${cls}: <strong>${count}</strong></span>`;
  }
  document.getElementById('stats').innerHTML = html;
  
  // Monta botões de filtro
  let filtersHtml = '<button class="active" data-class="all">Todas</button>';
  for (let i = 0; i < classNames.length; i++) {
    filtersHtml += `<button data-class="${i}">${classNames[i]}</button>`;
  }
  document.getElementById('filters').innerHTML = filtersHtml;
  
  document.querySelectorAll('.filters button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const cls = btn.dataset.class;
      currentFilter = cls === 'all' ? null : cls;
      resetAndLoad();
    });
  });
}

async function loadBatch() {
  if (loading || !hasMore) return;
  loading = true;
  document.getElementById('loader').textContent = 'Carregando...';
  
  let url = `/api/images?offset=${offset}&limit=${BATCH_SIZE}`;
  if (currentFilter !== null) url += `&class=${currentFilter}`;
  
  try {
    const res = await fetch(url);
    const data = await res.json();
    
    const grid = document.getElementById('grid');
    for (const item of data.items) {
      const card = document.createElement('div');
      card.className = 'card';
      card.onclick = () => openModal(item.filename);
      
      let badges = '';
      for (const cls of item.classes) {
        const name = classNames[cls] || `class_${cls}`;
        badges += `<span class="badge badge-${cls}">${name}</span>`;
      }
      
      card.innerHTML = `
        <img loading="lazy" src="/thumb/${encodeURIComponent(item.filename)}" alt="${item.filename}">
        <div class="card-info">
          <div class="card-name" title="${item.filename}">${item.filename}</div>
          <div class="card-badges">
            ${item.n_boxes === 0 ? '<span style="color:#888">sem anotações</span>' : badges}
            <span style="color:#666;margin-left:auto">${item.n_boxes} box${item.n_boxes!==1?'es':''}</span>
          </div>
        </div>
      `;
      grid.appendChild(card);
    }
    
    offset += data.items.length;
    hasMore = data.has_more;
    
    if (!hasMore) {
      document.getElementById('loader').textContent = '✓ Todas as imagens carregadas';
    } else {
      document.getElementById('loader').textContent = '';
    }
  } catch (e) {
    document.getElementById('loader').textContent = '❌ Erro ao carregar: ' + e;
  } finally {
    loading = false;
  }
}

function resetAndLoad() {
  offset = 0;
  hasMore = true;
  document.getElementById('grid').innerHTML = '';
  loadBatch();
}

// IntersectionObserver para infinite scroll
const observer = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) loadBatch();
}, { rootMargin: '400px' });

observer.observe(document.getElementById('sentinel'));

// Modal
function openModal(filename) {
  const modal = document.getElementById('modal');
  document.getElementById('modal-img').src = `/full/${encodeURIComponent(filename)}`;
  document.getElementById('modal-info').textContent = filename;
  modal.classList.add('active');
}
function closeModal() {
  document.getElementById('modal').classList.remove('active');
  document.getElementById('modal-img').src = '';
}
document.getElementById('modal').addEventListener('click', (e) => {
  if (e.target.id === 'modal') closeModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// Inicializa
loadStats().then(() => loadBatch());
</script>
</body>
</html>
"""


# ========== MAIN ==========

def main():
    global DATASET_DIR, IMAGES_DIR, LABELS_DIR, THUMBS_DIR
    
    parser = argparse.ArgumentParser(description='Visualizador de dataset YOLO')
    parser.add_argument('--dataset', type=str, default='./dataset_rotulado',
                        help='Caminho do dataset (padrão: ./dataset_rotulado)')
    parser.add_argument('--images', type=str, default=None,
                        help='Subpasta de imagens (padrão: images/train)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Subpasta de labels (padrão: labels/train)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Porta do servidor (padrão: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host (padrão: 0.0.0.0 para acesso remoto)')
    parser.add_argument('--classes', type=str, default='ambulance,car',
                        help='Nomes das classes separados por vírgula')
    
    args = parser.parse_args()
    
    # Configura nomes de classes
    global CLASS_NAMES
    CLASS_NAMES = [c.strip() for c in args.classes.split(',')]
    
    # Configura paths
    DATASET_DIR = Path(args.dataset)
    if not DATASET_DIR.exists():
        print(f"❌ Dataset não encontrado: {DATASET_DIR}")
        return
    
    # Tenta detectar estrutura automaticamente
    IMAGES_DIR = Path(args.images) if args.images else DATASET_DIR / 'images' / 'train'
    LABELS_DIR = Path(args.labels) if args.labels else DATASET_DIR / 'labels' / 'train'
    
    # Se não achou train/, tenta sem subpasta
    if not IMAGES_DIR.exists():
        IMAGES_DIR = DATASET_DIR / 'images'
    if not LABELS_DIR.exists():
        LABELS_DIR = DATASET_DIR / 'labels'
    
    THUMBS_DIR = DATASET_DIR / '.thumbs_cache'
    
    print("=" * 60)
    print("🖼️  VISUALIZADOR DE DATASET")
    print("=" * 60)
    print(f"📁 Dataset:  {DATASET_DIR.absolute()}")
    print(f"🖼️  Imagens:  {IMAGES_DIR}")
    print(f"🏷️  Labels:   {LABELS_DIR}")
    print(f"💾 Thumbs:   {THUMBS_DIR}")
    print(f"🏷️  Classes:  {CLASS_NAMES}")
    print("=" * 60)
    
    if not IMAGES_DIR.exists():
        print(f"❌ Pasta de imagens não encontrada: {IMAGES_DIR}")
        return
    
    # Pré-carrega lista
    get_image_list()
    
    print(f"\n🌐 Servidor rodando em:")
    print(f"   Local:   http://localhost:{args.port}")
    print(f"   Remoto:  http://0.0.0.0:{args.port}")
    print(f"\n💡 Acesse pelo browser da sua máquina local usando o IP remoto")
    print(f"   Exemplo: http://IP_DA_MAQUINA:{args.port}")
    print(f"\n   Ctrl+C para parar\n")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
