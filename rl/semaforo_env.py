import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import cv2
import random
from ultralytics import YOLO

class SemaforoInteligenteEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. ESPAÇOS DE AÇÃO E OBSERVAÇÃO
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([50, 50, 50, 300.0, 300.0, 300.0, 1.0, 150.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.estado_atual = np.zeros(9, dtype=np.float32)
        self.ciclos_atuais = 0
        self.max_ciclos = 200
        self.esp_34 = 0.0
        self.esp_152 = 0.0
        self.esp_92 = 0.0
        
        # 2. INTEGRAÇÃO YOLO
        print("[SISTEMA] Carregando YOLO...")
        self.yolo = YOLO("yolov8n.pt") 
        self.frame_atual = None
        
        # 3. INTEGRAÇÃO CARLA (Conexão e Modo Síncrono)
        print("[SISTEMA] Conectando ao CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05 # 20 FPS exatos para curvas perfeitas
        self.world.apply_settings(self.settings)

        # MÁGICA: O Traffic Manager agora vive no Cérebro do RL e 100% síncrono!
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(1.5)
        self.tm.set_random_device_seed(0)

        # 4. MAPEAMENTO DOS SEMÁFOROS
        coord_refs = {
            0: carla.Location(x=324.9, y=332.7, z=3.0),
            1: carla.Location(x=350.2, y=324.3, z=3.0),
            2: carla.Location(x=332.1, y=316.2, z=3.0) 
        }
        self.semaforos = {0: None, 1: None, 2: None}
        
        todos_semaforos = self.world.get_actors().filter('traffic.traffic_light')
        for acao_id, coord in coord_refs.items():
            mais_prox = min(todos_semaforos, key=lambda s: s.get_location().distance(coord))
            self.semaforos[acao_id] = mais_prox
            mais_prox.freeze(True)
            mais_prox.set_state(carla.TrafficLightState.Red)

        # 5. RETÂNGULOS VIRTUAIS
        self.areas_de_contagem = {
            1: {'x_min': 355.0, 'x_max': 387.0, 'y_min': 325.5, 'y_max': 327.5},
            2: {'x_min': 334.0, 'x_max': 336.0, 'y_min': 240.0, 'y_max': 314.0} 
        }

        # 6. INSTANCIAR CÂMERA DO CARLA
        self.bp_lib = self.world.get_blueprint_library()
        cam_bp = self.bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        transform_camera = carla.Transform(
            carla.Location(x=323.0, y=320.0, z=3.0), 
            carla.Rotation(pitch=0.0, yaw=180.0)
        )
        self.camera = self.world.spawn_actor(cam_bp, transform_camera)
        self.camera.listen(lambda data: self._processar_imagem_camera(data))

        # 7. SETUP DO TRÁFEGO DENTRO DO RL
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.bp_carros = [
            self.bp_lib.find('vehicle.audi.a2'), self.bp_lib.find('vehicle.audi.etron'),
            self.bp_lib.find('vehicle.audi.tt'), self.bp_lib.find('vehicle.bmw.grandtourer'),
            self.bp_lib.find('vehicle.chevrolet.impala'), self.bp_lib.find('vehicle.citroen.c3'),
            self.bp_lib.find('vehicle.ford.crown'), self.bp_lib.find('vehicle.ford.mustang'),
            self.bp_lib.find('vehicle.mercedes.coupe'), self.bp_lib.find('vehicle.micro.microlino'),
            self.bp_lib.find('vehicle.nissan.micra'), self.bp_lib.find('vehicle.nissan.patrol'),
            self.bp_lib.find('vehicle.seat.leon'), self.bp_lib.find('vehicle.tesla.model3'),
            self.bp_lib.find('vehicle.toyota.prius'), self.bp_lib.find('vehicle.volkswagen.t2'),
            self.bp_lib.find('vehicle.kawasaki.ninja'), self.bp_lib.find('vehicle.vespa.zx125'),
            self.bp_lib.find('vehicle.yamaha.yzf'), self.bp_lib.find('vehicle.bh.crossbike'),
            self.bp_lib.find('vehicle.diamondback.century'), self.bp_lib.find('vehicle.gazelle.omafiets')
        ]
        self.bp_ambulancia = self.bp_lib.find('vehicle.ford.ambulance')
        self.lista_atores = []
        self.contador_frames_globais = 0
        self.ambulancia_pendente = False

    def _processar_imagem_camera(self, imagem):
        """Callback do CARLA - Apenas salva a matriz, SEM desenhar tela aqui!"""
        array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (imagem.height, imagem.width, 4))
        self.frame_atual = array[:, :, :3]

    def _extrair_estado_da_imagem(self):
        """O Cérebro Visual Híbrido (Roda na Thread Principal)"""
        fila_152 = self._contar_fila_simulada(1)
        fila_92  = self._contar_fila_simulada(2)
        
        fila_34 = 0
        tem_amb, conf_amb, dist_amb = 0.0, 0.0, 0.0
        
        if self.frame_atual is not None:
            # === A JANELA SEGURA FICA AQUI ===
            # cv2.imshow("Cerebro do RL - Câmera YOLO", self.frame_atual)
            # cv2.waitKey(1)
            
            # Inferência da YOLO
            resultados = self.yolo(self.frame_atual, verbose=False)[0]
            
            for box in resultados.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 2: 
                    fila_34 += 1
                elif cls_id == 7: 
                    tem_amb = 1.0
                    conf_amb = conf
                    dist_amb = 80.0 

        self.esp_34  += 5.0 if fila_34 > 0 else -self.esp_34
        self.esp_152 += 5.0 if fila_152 > 0 else -self.esp_152
        self.esp_92  += 5.0 if fila_92 > 0 else -self.esp_92
        
        self.esp_34 = max(0.0, self.esp_34)
        self.esp_152 = max(0.0, self.esp_152)
        self.esp_92 = max(0.0, self.esp_92)

        return np.array([fila_34, fila_152, fila_92, 
                         self.esp_34, self.esp_152, self.esp_92, 
                         tem_amb, dist_amb, conf_amb], dtype=np.float32)

    def _contar_fila_simulada(self, id_via):
        area = self.areas_de_contagem[id_via]
        veiculos = self.world.get_actors().filter('vehicle.*')
        fila = 0
        for v in veiculos:
            loc = v.get_location()
            if (area['x_min'] <= loc.x <= area['x_max'] and 
                area['y_min'] <= loc.y <= area['y_max']):
                fila += 1
        return fila

    def _gerenciar_trafego(self):
        """Mágica do Tráfego: Executa a cada frame perfeitamente sincronizado com o RL"""
        self.contador_frames_globais += 1
        veiculos_ativos = [a for a in self.lista_atores if a.is_alive]
        ambulancias_ativas = [a for a in veiculos_ativos if 'ambulance' in a.type_id]
        
        # Gatilho da Ambulância
        if self.contador_frames_globais > 0 and self.contador_frames_globais % 800 == 0:
            if len(ambulancias_ativas) == 0 and not self.ambulancia_pendente:
                self.ambulancia_pendente = True

        if self.ambulancia_pendente:
            amb = self.world.try_spawn_actor(self.bp_ambulancia, self.spawn_points[34])
            if amb is not None:
                amb.set_simulate_physics(True)
                amb.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                amb.set_target_velocity(self.spawn_points[34].get_forward_vector() * 7.0)
                amb.set_autopilot(True, self.tm.get_port())
                
                self.tm.ignore_lights_percentage(amb, 0.0)
                self.tm.auto_lane_change(amb, False)
                self.tm.set_desired_speed(amb, 12.0)
                self.tm.set_route(amb, ["Straight", "Straight"]) 
                
                self.lista_atores.append(amb)
                self.ambulancia_pendente = False                   

        # Gatilho de Carros Comuns (O BUG ESTAVA AQUI)
        if self.contador_frames_globais % 8 == 0 and len(veiculos_ativos) < 120:
            id_sorteado = random.choice([34, 152, 92])
            
            # REMOVIDO O BLOQUEIO DA VIA 34. Agora o sorteio é livre e justo.
                
            carro = self.world.try_spawn_actor(random.choice(self.bp_carros), self.spawn_points[id_sorteado])
            if carro is not None:
                carro.set_simulate_physics(True)
                carro.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                carro.set_target_velocity(self.spawn_points[id_sorteado].get_forward_vector() * 5.0)
                carro.set_autopilot(True, self.tm.get_port())
                
                self.tm.ignore_lights_percentage(carro, 0.0)
                self.tm.auto_lane_change(carro, False)
                self.tm.set_desired_speed(carro, 12.0) 
                self.lista_atores.append(carro)

        # Limpeza de Despawn
        posicao_cruzamento = carla.Location(x=323.0, y=320.0, z=2.0)
        for ator in self.lista_atores[:]:
            try:
                if ator.is_alive:
                    loc = ator.get_location()
                    if loc.distance(posicao_cruzamento) > 150.0 or loc.z < -10.0:
                        ator.destroy()
                        if ator in self.lista_atores:
                            self.lista_atores.remove(ator)
            except RuntimeError:
                if ator in self.lista_atores:
                    self.lista_atores.remove(ator)

    def step(self, action):
        self.ciclos_atuais += 1
        
        for i in range(3):
            if self.semaforos[i]:
                self.semaforos[i].set_state(carla.TrafficLightState.Red)
        if self.semaforos[action]:
            self.semaforos[action].set_state(carla.TrafficLightState.Green)
            
        # O RL avança o tempo e atualiza a sua janela a 20 FPS perfeitos!
        for _ in range(100):
            self._gerenciar_trafego()
            self.world.tick() 
            
            # --- A JANELA SEGURA E FLUIDA ESTÁ AQUI ---
            if self.frame_atual is not None:
                cv2.imshow("Monitoramento de Trafego - RL", self.frame_atual)
                cv2.waitKey(1)
            
        novo_estado = self._extrair_estado_da_imagem() 
        reward = self._calcular_recompensa(novo_estado, action)
        
        terminated = False
        truncated = bool(self.ciclos_atuais >= self.max_ciclos)
        
        self.estado_atual = novo_estado
        return novo_estado, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ciclos_atuais = 0
        self.esp_34 = self.esp_152 = self.esp_92 = 0.0
        
        for ator in self.lista_atores:
            try:
                if ator.is_alive:
                    ator.destroy()
            except RuntimeError:
                pass 
        self.lista_atores.clear()
        self.contador_frames_globais = 0
        
        for _ in range(10): 
            self._gerenciar_trafego()
            self.world.tick()
            # Mostra os frames iniciais também
            if self.frame_atual is not None:
                cv2.imshow("Monitoramento de Trafego - RL", self.frame_atual)
                cv2.waitKey(1)
            
        self.estado_atual = self._extrair_estado_da_imagem()
        return self.estado_atual, {}

    def _calcular_recompensa(self, estado, acao):
        fila_34, fila_152, fila_92, esp_34, esp_152, esp_92, tem_amb, dist_amb, conf_amb = estado
        reward = 0.0
        
        penalidade_espera = ((esp_34**2) + (esp_152**2) + (esp_92**2)) * 0.001
        penalidade_filas = (fila_34 + fila_152 + fila_92) * 0.5
        reward -= (penalidade_espera + penalidade_filas)
        
        if tem_amb == 1.0:
            if acao != 0:
                fator_distancia = 500.0 / (dist_amb + 1.0)
                fator_escoamento = fila_34 * 10.0
                reward -= (fator_distancia + fator_escoamento) * conf_amb
            else:
                reward += (50.0 * conf_amb)
                
        return float(reward)

    def close(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            
        for ator in self.lista_atores:
            try:
                if ator.is_alive:
                    ator.destroy()
            except RuntimeError:
                pass
                
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        cv2.destroyAllWindows()