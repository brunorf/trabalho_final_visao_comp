import os
import cv2
import base64
from glob import glob
from flask import Flask, render_template_string, request

app = Flask(__name__)

# 1. Configuração do caminho do dataset
PASTA_BASE = "/home/bruno.rogerio/.cache/kagglehub/datasets/mahmoudshaheen1134/ambulance-dataset/versions/1/carsdataset"
EXTENSOES = ('*.jpg', '*.png', '*.jpeg')

print("Mapeando imagens no dataset...")
todas_imagens = []
for ext in EXTENSOES:
    # O '**' com recursive=True garante que ele entre nas pastas ambulance e noambulance
    caminho_busca = os.path.join(PASTA_BASE, '**', ext)
    todas_imagens.extend(glob(caminho_busca, recursive=True))

todas_imagens.sort()
print(f"Total de {len(todas_imagens)} imagens encontradas.")

# 2. Template HTML com CSS embutido para a grade e paginação
TEMPLATE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Validador de Dataset YOLO</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #1e1e1e; color: #fff; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .card { background: #2d2d2d; padding: 10px; border-radius: 8px; border: 1px solid #444; }
        img { max-width: 100%; height: auto; border-radius: 4px; }
        .caminho { font-size: 11px; word-break: break-all; color: #aaa; margin-top: 10px; }
        .erro { color: #ff4444; font-weight: bold; font-size: 14px; }
        .pagination { margin: 30px 0; text-align: center; }
        .btn { padding: 10px 20px; margin: 0 5px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
        .btn.disabled { background: #555; pointer-events: none; color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Verificação de Labels - Página {{ pagina }} de {{ total_paginas }}</h2>
        <p>Total do Dataset: {{ total_imagens }} imagens</p>
    </div>
    
    <div class="pagination">
        <a href="/?pagina={{ pagina - 1 }}" class="btn {% if pagina <= 1 %}disabled{% endif %}">Anterior</a>
        <a href="/?pagina={{ pagina + 1 }}" class="btn {% if pagina >= total_paginas %}disabled{% endif %}">Próxima</a>
    </div>

    <div class="grid">
        {% for img in imagens %}
        <div class="card">
            <img src="data:image/jpeg;base64,{{ img.base64 }}">
            <div class="caminho">{{ img.caminho }}</div>
        </div>
        {% endfor %}
    </div>

    <div class="pagination">
        <a href="/?pagina={{ pagina - 1 }}" class="btn {% if pagina <= 1 %}disabled{% endif %}">Anterior</a>
        <a href="/?pagina={{ pagina + 1 }}" class="btn {% if pagina >= total_paginas %}disabled{% endif %}">Próxima</a>
    </div>
</body>
</html>
"""

def desenhar_labels(caminho_imagem):
    """Lê a imagem e o .txt, desenha as caixas e retorna em formato base64"""
    img = cv2.imread(caminho_imagem)
    if img is None:
        return None
    
    altura, largura = img.shape[:2]
    
    # Assume que o .txt tem o mesmo nome da imagem e está na mesma pasta
    caminho_label = os.path.splitext(caminho_imagem)[0] + '.txt'
    
    if os.path.exists(caminho_label):
        with open(caminho_label, 'r') as f:
            linhas = f.readlines()
        
        for linha in linhas:
            dados = linha.strip().split()
            if len(dados) >= 5:
                classe = int(dados[0])
                x_center, y_center, w, h = map(float, dados[1:5])
                
                # Desnormalização
                x_min = int((x_center - w/2) * largura)
                y_min = int((y_center - h/2) * altura)
                x_max = int((x_center + w/2) * largura)
                y_max = int((y_center + h/2) * altura)
                
                # Cores BGR: Verde para Ambulância (1), Vermelho para Carro (0)
                cor = (0, 255, 0) if classe == 1 else (0, 0, 255) 
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cor, 2)
                cv2.putText(img, f"ID: {classe}", (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
    else:
        # Alerta visual caso a imagem não tenha arquivo .txt associado
        cv2.putText(img, "SEM LABEL (.txt)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # Converte a imagem processada para base64 para ser exibida diretamente no HTML
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    # Lógica de Paginação
    pagina = int(request.args.get('pagina', 1))
    por_pagina = 20
    
    total_imagens = len(todas_imagens)
    total_paginas = (total_imagens + por_pagina - 1) // por_pagina
    
    inicio = (pagina - 1) * por_pagina
    fim = inicio + por_pagina
    
    imagens_processadas = []
    
    # Processa apenas as 20 imagens da página atual
    for caminho in todas_imagens[inicio:fim]:
        img_b64 = desenhar_labels(caminho)
        if img_b64:
            imagens_processadas.append({
                'caminho': caminho, 
                'base64': img_b64
            })
            
    return render_template_string(
        TEMPLATE_HTML, 
        imagens=imagens_processadas, 
        pagina=pagina, 
        total_paginas=total_paginas,
        total_imagens=total_imagens
    )

if __name__ == '__main__':
    print("\nServidor iniciado! Abra o navegador e acesse: http://127.0.0.1:8080\n")
    app.run(host='0.0.0.0', port=8080, debug=False)
