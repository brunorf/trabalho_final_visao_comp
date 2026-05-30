import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import cv2
from ultralytics import YOLO

class SemaforoInteligenteEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. ESPAÇOS DE AÇÃO E OBSERVAÇÃO
        self.action_space = spaces.Discrete(3)
        # [fila_34, fila_152, fila_92, esp_34, esp_152, esp_92, tem_amb, dist_amb, conf_amb]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([50, 50, 50, 300.0, 300.0, 300.0, 1.0, 150.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.estado_atual = np.zeros(9, dtype=np.float32)
        self.ciclos_atuais = 0
        self.max_ciclos = 200
        
        # Tempos de espera acumulados
        self.esp_34 = 0.0
        self.esp_152 = 0.0
        self.esp_92 = 0.0
        
        # 2. INTEGRAÇÃO YOLO (Carrega na memória uma única vez)
        print("[SISTEMA] Carregando YOLO...")
        self.yolo = YOLO("yolov8n.pt") # TODO: Mudar para o seu arquivo .pt treinado
        self.frame_atual = None
        
        # 3. INTEGRAÇÃO CARLA (Conexão e Modo Síncrono)
        print("[SISTEMA] Conectando ao CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        
        # ---> ALTERE O VALOR AQUI PARA 0.05 <---
        self.settings.fixed_delta_seconds = 0.05 
        
        self.world.apply_settings(self.settings)

        # 4. MAPEAMENTO DOS SEMÁFOROS (As suas coordenadas da Fase 1)
        coord_refs = {
            0: carla.Location(x=324.9, y=332.7, z=3.0), # Via 34
            1: carla.Location(x=350.2, y=324.3, z=3.0), # Via 152
            2: carla.Location(x=332.1, y=316.2, z=3.0)  # Via 92
        }
        self.semaforos = {0: None, 1: None, 2: None}
        
        todos_semaforos = self.world.get_actors().filter('traffic.traffic_light')
        for acao_id, coord in coord_refs.items():
            mais_prox = min(todos_semaforos, key=lambda s: s.get_location().distance(coord))
            self.semaforos[acao_id] = mais_prox
            mais_prox.freeze(True)
            mais_prox.set_state(carla.TrafficLightState.Red)

        # 5. RETÂNGULOS VIRTUAIS (As coordenadas perfeitas que você mediu)
        self.areas_de_contagem = {
            1: {'x_min': 355.0, 'x_max': 387.0, 'y_min': 325.5, 'y_max': 327.5}, # Via 152
            2: {'x_min': 334.0, 'x_max': 336.0, 'y_min': 240.0, 'y_max': 314.0}  # Via 92
        }

        # 6. INSTANCIAR CÂMERA DO CARLA
        bp_lib = self.world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        transform_camera = carla.Transform(
            carla.Location(x=323.0, y=320.0, z=3.0), 
            carla.Rotation(pitch=0.0, yaw=180.0)
        )
        self.camera = self.world.spawn_actor(cam_bp, transform_camera)
        self.camera.listen(lambda data: self._processar_imagem_camera(data))

    def _processar_imagem_camera(self, imagem):
        """Callback do CARLA que atualiza o self.frame_atual em BGR (OpenCV)"""
        array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (imagem.height, imagem.width, 4))
        self.frame_atual = array[:, :, :3]

    def _contar_fila_simulada(self, id_via):
        """Usa seus retângulos virtuais para contar carros nas vias 152 e 92"""
        area = self.areas_de_contagem[id_via]
        veiculos = self.world.get_actors().filter('vehicle.*')
        fila = 0
        for v in veiculos:
            loc = v.get_location()
            if (area['x_min'] <= loc.x <= area['x_max'] and 
                area['y_min'] <= loc.y <= area['y_max']):
                fila += 1
        return fila

    def _extrair_estado_da_imagem(self):
        """O Cérebro Visual Híbrido (Roda após o CARLA avançar o tempo)"""
        
        # Filas simuladas pelas suas caixas delimitadoras
        fila_152 = self._contar_fila_simulada(1)
        fila_92  = self._contar_fila_simulada(2)
        
        fila_34 = 0
        tem_amb, conf_amb, dist_amb = 0.0, 0.0, 0.0
        
        if self.frame_atual is not None:
            # Roda a YOLO apenas na imagem atual (sem travar por excesso de frames)
            resultados = self.yolo(self.frame_atual, verbose=False)[0]
            
            for box in resultados.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Se for carro comum na visão da câmera (Via 34)
                if cls_id == 2: # TODO: Ajustar para o ID de 'car' no seu modelo
                    fila_34 += 1
                
                # Se for ambulância
                elif cls_id == 7: # TODO: Ajustar para o ID de 'truck' (sua ambulância)
                    tem_amb = 1.0
                    conf_amb = conf
                    # Placeholder para o apple ml-depth-pro (que você vai plugar depois)
                    dist_amb = 80.0 

        # Atualiza a matemática de tempo de espera (Starvation)
        self.esp_34  += 5.0 if fila_34 > 0 else -self.esp_34
        self.esp_152 += 5.0 if fila_152 > 0 else -self.esp_152
        self.esp_92  += 5.0 if fila_92 > 0 else -self.esp_92
        
        self.esp_34 = max(0.0, self.esp_34)
        self.esp_152 = max(0.0, self.esp_152)
        self.esp_92 = max(0.0, self.esp_92)

        return np.array([fila_34, fila_152, fila_92, 
                         self.esp_34, self.esp_152, self.esp_92, 
                         tem_amb, dist_amb, conf_amb], dtype=np.float32)

    def step(self, action):
        self.ciclos_atuais += 1
        
        # 1. Aplica a cor do semáforo
        for i in range(3):
            if self.semaforos[i]:
                self.semaforos[i].set_state(carla.TrafficLightState.Red)
        if self.semaforos[action]:
            self.semaforos[action].set_state(carla.TrafficLightState.Green)
            
        # 2. Avança o trânsito em 5 segundos simulados (50 ticks de 0.1s)
        for _ in range(100):
            self.world.tick()
            
        # 3. Lê o mundo (Ground Truth + YOLO)
        novo_estado = self._extrair_estado_da_imagem() 
        
        # 4. Calcula o Reward
        reward = self._calcular_recompensa(novo_estado, action)
        
        terminated = False
        truncated = bool(self.ciclos_atuais >= self.max_ciclos)
        
        self.estado_atual = novo_estado
        return novo_estado, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ciclos_atuais = 0
        self.esp_34 = self.esp_152 = self.esp_92 = 0.0
        
        # Deixa a simulação estabilizar (câmera gerar o primeiro frame)
        for _ in range(10): 
            self.world.tick()
            
        self.estado_atual = self._extrair_estado_da_imagem()
        return self.estado_atual, {}

    def _calcular_recompensa(self, estado, acao):
        fila_34, fila_152, fila_92, esp_34, esp_152, esp_92, tem_amb, dist_amb, conf_amb = estado
        reward = 0.0
        
        # Punição quadrática para evitar starvation
        penalidade_espera = ((esp_34**2) + (esp_152**2) + (esp_92**2)) * 0.001
        penalidade_filas = (fila_34 + fila_152 + fila_92) * 0.5
        reward -= (penalidade_espera + penalidade_filas)
        
        # Prioridade com Fallback da YOLO para ambulância
        if tem_amb == 1.0:
            if acao != 0:
                fator_distancia = 500.0 / (dist_amb + 1.0)
                fator_escoamento = fila_34 * 10.0
                reward -= (fator_distancia + fator_escoamento) * conf_amb
            else:
                reward += (50.0 * conf_amb)
                
        return float(reward)

    def close(self):
        # Limpeza para quando o treino acabar
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)