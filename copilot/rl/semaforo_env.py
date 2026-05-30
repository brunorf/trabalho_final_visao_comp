import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import random
import sys
import math
import json
import os

class SemaforoInteligenteEnv(gym.Env):
    def __init__(self, host_carla='127.0.0.1', porta_carla=2000, porta_tm=8000):
        super().__init__()
        self.ultima_acao = -1
        self.acao_atual = 0

        # 1. ESPAÇOS DE AÇÃO E OBSERVAÇÃO
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50, 50, 50, 300.0, 300.0, 300.0, 1.0, 150.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.estado_atual = np.zeros(9, dtype=np.float32)
        self.ciclos_atuais = 0
        self.max_ciclos = 1000

        self.espera_via_ambulancia = 0.0
        self.espera_via_oposta = 0.0
        self.espera_via_perpendicular = 0.0

        # 2. INTEGRAÇÃO CARLA
        print("[SISTEMA] Conectando ao CARLA...")
        self.client = carla.Client(host_carla, porta_carla)
        self.client.set_timeout(60.0)

        try:
            self.world = self.client.get_world()
            if not self.world.get_map().name.endswith('Town01'):
                print("Carregando Town01...")
                self.world = self.client.load_world('Town01')
        except Exception as e:
            print(f"Erro ao conectar/carregar mundo: {e}")
            sys.exit(1)

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        # Traffic Manager (Configurado para gerar filas reais)
        self.tm = self.client.get_trafficmanager(porta_tm)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(0)
        self.tm.set_global_distance_to_leading_vehicle(1.5)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(50.0)

        self.max_veiculos_simultaneos = 120
        self.ambulancia_pendente = False
        self.ambulancias_ja_passaram = set()

        self.bp_lib = self.world.get_blueprint_library()

        # 3. Identidade semântica das vias e fallback legado de índice de spawn
        self.VIA_AMBULANCIA = "via_ambulancia"
        self.VIA_OPOSTA = "via_oposta"
        self.VIA_PERPENDICULAR = "via_perpendicular"
        self.vias = [self.VIA_AMBULANCIA, self.VIA_OPOSTA, self.VIA_PERPENDICULAR]

        self.ACAO_VIA_AMBULANCIA = 0
        self.ACAO_VIA_OPOSTA = 1
        self.ACAO_VIA_PERPENDICULAR = 2
        self.via_por_acao = [self.VIA_AMBULANCIA, self.VIA_OPOSTA, self.VIA_PERPENDICULAR]

        self.legacy_spawn_index_por_via = {
            self.VIA_AMBULANCIA: 34,
            self.VIA_OPOSTA: 152,
            self.VIA_PERPENDICULAR: 92,
        }

        # 4. Geometria global
        self.posicao_cruzamento = carla.Location(x=323.0, y=320.0, z=2.0)
        self.area_cruzamento = {
            'x_min': 315.0, 'x_max': 335.0,
            'y_min': 310.0, 'y_max': 330.0
        }

        # 5. Spawns por referência (com fallback para IDs)
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.pontos_spawn = self._resolver_spawns_por_referencia_ou_id()

        dists = [
            self.pontos_spawn[self.VIA_AMBULANCIA].location.distance(self.posicao_cruzamento),
            self.pontos_spawn[self.VIA_OPOSTA].location.distance(self.posicao_cruzamento),
            self.pontos_spawn[self.VIA_PERPENDICULAR].location.distance(self.posicao_cruzamento),
        ]
        self.raio_despawn = max(dists) + 20.0

        # 6. Corredores dinâmicos por trilha de waypoints
        self.CORREDOR_LARGURA_MEIA = 6.0
        self.MARGEM_CRUZAMENTO = 8.0
        self.PASSO_WAYPOINT_METROS = 2.0
        self.MAX_PONTOS_TRILHA = 180
        self.PASSOS_EXTRA_APOS_CRUZAMENTO = 12
        self.BUFFER_FIM_VISAO_CAMERA_AMB = 3.0
        self.BUFFER_DESCARTE_AMB = 12.0
        self.LIMIAR_VELOCIDADE_FILA = 1.5

        self.trilhas_vias = {}
        self.chaves_lane_por_via = {}
        for via_id in self.vias:
            trilha, chaves = self._construir_trilha_ate_cruzamento(self.pontos_spawn[via_id])
            self.trilhas_vias[via_id] = trilha
            self.chaves_lane_por_via[via_id] = chaves

        self.s_cruzamento_por_via = {}
        for via_id in self.vias:
            _, s_cruz, _ = self._metrica_para_trilha(self.posicao_cruzamento, self.trilhas_vias[via_id])
            self.s_cruzamento_por_via[via_id] = s_cruz

        # 7. Mapeamento dos semáforos por via
        todos_semaforos = [
            tl for tl in self.world.get_actors().filter('traffic.traffic_light')
            if tl.get_location().distance(self.posicao_cruzamento) < 35.0
        ]
        self.semaforos = self._mapear_semaforos_por_via(todos_semaforos)
        for i in range(3):
            if self.semaforos[i]:
                self.semaforos[i].freeze(True)
                self.semaforos[i].set_state(carla.TrafficLightState.Red)

        # Tráfego fora do cruzamento local fica verde para reduzir interferência.
        for semaforo in todos_semaforos:
            if semaforo not in self.semaforos.values():
                semaforo.freeze(True)
                semaforo.set_state(carla.TrafficLightState.Green)

        # 8. Spawning e velocidade
        self.proximo_pulse = 0
        self.fila_de_pulsos = []
        self.frequencia_spawn_carros = 10

        self.VELOCIDADE_BASE_FECHADO = 4.0
        self.VELOCIDADE_BASE_ABERTO = 9.0
        self.VELOCIDADE_AMB_FECHADO = 5.0
        self.VELOCIDADE_AMB_ABERTO = 13.0
        self.FRAMES_ATUALIZACAO_VELOCIDADE = 10

        # 9. Blueprints e estado de episódio
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
        self.contador_frames = 0
        self.ticks_por_step = 20

    def _dist2d(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _distancia_ponto_segmento_2d(self, px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom < 1e-9:
            t = 0.0
            qx, qy = ax, ay
        else:
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
            qx = ax + t * abx
            qy = ay + t * aby
        d = math.hypot(px - qx, py - qy)
        return d, t

    def _spawn_transform_from_reference(self, ref_loc, metros_recuo=14.0):
        mapa = self.world.get_map()
        wp = mapa.get_waypoint(ref_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            raise RuntimeError(f"Nao foi possivel projetar referencia de spawn: {ref_loc}")
        passos = int(metros_recuo // 2)
        for _ in range(max(passos, 0)):
            prev = wp.previous(2.0)
            if not prev:
                break
            wp = prev[0]
        t = wp.transform
        t.location.z += 0.3
        return t

    def _carregar_referencias_spawn(self):
        candidatos = [
            os.path.join(os.getcwd(), "spawn_refs.json"),
            os.path.join(os.path.dirname(__file__), "spawn_refs.json"),
            os.path.join(os.path.dirname(__file__), "..", "spawn_refs.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "spawn_refs.json"),
        ]
        for path in candidatos:
            path = os.path.abspath(path)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                refs = data.get("references", {})
                # Formato novo: chaves semânticas.
                if all(via in refs for via in self.vias):
                    out = {
                        self.VIA_AMBULANCIA: carla.Location(**refs[self.VIA_AMBULANCIA]),
                        self.VIA_OPOSTA: carla.Location(**refs[self.VIA_OPOSTA]),
                        self.VIA_PERPENDICULAR: carla.Location(**refs[self.VIA_PERPENDICULAR]),
                    }
                    print(f"[CALIB] Referencias de spawn carregadas de: {path}")
                    return out

                # Formato antigo: chaves numéricas em string (34/152/92).
                legacy_keys = [str(self.legacy_spawn_index_por_via[via]) for via in self.vias]
                if not all(k in refs for k in legacy_keys):
                    continue
                out = {
                    self.VIA_AMBULANCIA: carla.Location(**refs[str(self.legacy_spawn_index_por_via[self.VIA_AMBULANCIA])]),
                    self.VIA_OPOSTA: carla.Location(**refs[str(self.legacy_spawn_index_por_via[self.VIA_OPOSTA])]),
                    self.VIA_PERPENDICULAR: carla.Location(**refs[str(self.legacy_spawn_index_por_via[self.VIA_PERPENDICULAR])]),
                }
                print(f"[CALIB] Referencias de spawn carregadas de: {path}")
                return out
            except Exception as e:
                print(f"[CALIB] Falha ao ler {path}: {e}")
        return None

    def _resolver_spawns_por_referencia_ou_id(self):
        referencias_padrao = {
            self.VIA_AMBULANCIA: carla.Location(x=322.5, y=280.0, z=0.5),
            self.VIA_OPOSTA: carla.Location(x=322.5, y=365.0, z=0.5),
            self.VIA_PERPENDICULAR: carla.Location(x=280.0, y=322.5, z=0.5),
        }
        referencias = self._carregar_referencias_spawn() or referencias_padrao
        try:
            return {via: self._spawn_transform_from_reference(referencias[via]) for via in self.vias}
        except Exception as e:
            print(f"[SPAWN] Falha no modo referencia ({e}). Fallback para IDs fixos.")
            if max(self.legacy_spawn_index_por_via.values()) >= len(self.spawn_points):
                raise RuntimeError("IDs de spawn inválidos para o mapa atual")
            return {
                self.VIA_AMBULANCIA: self.spawn_points[self.legacy_spawn_index_por_via[self.VIA_AMBULANCIA]],
                self.VIA_OPOSTA: self.spawn_points[self.legacy_spawn_index_por_via[self.VIA_OPOSTA]],
                self.VIA_PERPENDICULAR: self.spawn_points[self.legacy_spawn_index_por_via[self.VIA_PERPENDICULAR]],
            }

    def _construir_trilha_ate_cruzamento(self, spawn_transform):
        mapa = self.world.get_map()
        wp = mapa.get_waypoint(spawn_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return [spawn_transform.location, self.posicao_cruzamento], set()

        trilha = [wp.transform.location]
        chaves_via = {(wp.road_id, wp.lane_id)}
        visitados = {(wp.road_id, wp.lane_id, round(wp.s, 1))}
        chegou_cruzamento = False
        passos_pos = 0

        for _ in range(self.MAX_PONTOS_TRILHA):
            if self._dist2d(wp.transform.location, self.posicao_cruzamento) <= self.MARGEM_CRUZAMENTO:
                chegou_cruzamento = True
            if chegou_cruzamento and passos_pos >= self.PASSOS_EXTRA_APOS_CRUZAMENTO:
                break

            candidatos = wp.next(self.PASSO_WAYPOINT_METROS)
            if not candidatos:
                break
            prox = min(candidatos, key=lambda c: self._dist2d(c.transform.location, self.posicao_cruzamento))
            chave = (prox.road_id, prox.lane_id, round(prox.s, 1))
            if chave in visitados:
                break

            visitados.add(chave)
            wp = prox
            trilha.append(wp.transform.location)
            chaves_via.add((wp.road_id, wp.lane_id))
            if chegou_cruzamento:
                passos_pos += 1

        if len(trilha) == 1:
            trilha.append(self.posicao_cruzamento)
        return trilha, chaves_via

    def _metrica_para_trilha(self, loc, trilha):
        if len(trilha) < 2:
            d = self._dist2d(loc, trilha[0]) if trilha else float('inf')
            return d, 0.0, 0.0

        px, py = loc.x, loc.y
        melhor_d = float('inf')
        melhor_s = 0.0
        s_acum = 0.0

        for i in range(len(trilha) - 1):
            a = trilha[i]
            b = trilha[i + 1]
            seg_len = self._dist2d(a, b)
            d, t = self._distancia_ponto_segmento_2d(px, py, a.x, a.y, b.x, b.y)
            if d < melhor_d:
                melhor_d = d
                melhor_s = s_acum + t * seg_len
            s_acum += seg_len

        return melhor_d, melhor_s, s_acum

    def _esta_no_corredor_da_via(self, loc, via_id):
        trilha = self.trilhas_vias[via_id]
        lateral, progresso_s, comp_total = self._metrica_para_trilha(loc, trilha)
        if progresso_s < 0.0 or progresso_s > (comp_total + self.MARGEM_CRUZAMENTO):
            return False
        return lateral <= self.CORREDOR_LARGURA_MEIA

    def _classificar_via_veiculo(self, loc):
        wp_v = self.world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_v is None:
            return None
        chave_v = (wp_v.road_id, wp_v.lane_id)
        candidatos = [i for i in self.vias if chave_v in self.chaves_lane_por_via[i]]
        if not candidatos:
            return None

        melhor = None
        melhor_lateral = float('inf')
        for via_id in candidatos:
            if not self._esta_no_corredor_da_via(loc, via_id):
                continue
            lateral, _, _ = self._metrica_para_trilha(loc, self.trilhas_vias[via_id])
            if lateral < melhor_lateral:
                melhor_lateral = lateral
                melhor = via_id
        return melhor

    def _ambulancia_visivel_na_camera(self, loc):
        trilha = self.trilhas_vias[self.VIA_AMBULANCIA]
        lateral, progresso_s, comp_total = self._metrica_para_trilha(loc, trilha)
        s_cruz = self.s_cruzamento_por_via[self.VIA_AMBULANCIA]
        if lateral > self.CORREDOR_LARGURA_MEIA or progresso_s < 0.0:
            return False
        limite_saida = min(comp_total, s_cruz + self.BUFFER_FIM_VISAO_CAMERA_AMB)
        return progresso_s <= limite_saida

    def _mapear_semaforos_por_via(self, semaforos):
        mapeados = {0: None, 1: None, 2: None}
        restantes = semaforos[:]
        origem_por_acao = {
            self.ACAO_VIA_AMBULANCIA: self.pontos_spawn[self.VIA_AMBULANCIA].location,
            self.ACAO_VIA_OPOSTA: self.pontos_spawn[self.VIA_OPOSTA].location,
            self.ACAO_VIA_PERPENDICULAR: self.pontos_spawn[self.VIA_PERPENDICULAR].location,
        }
        for acao in [self.ACAO_VIA_AMBULANCIA, self.ACAO_VIA_OPOSTA, self.ACAO_VIA_PERPENDICULAR]:
            if not restantes:
                break
            escolhido = min(restantes, key=lambda s: s.get_location().distance(origem_por_acao[acao]))
            mapeados[acao] = escolhido
            restantes.remove(escolhido)
        return mapeados

    def _try_spawn_with_offsets(self, bp, base_transform, offsets_m=(0.0, -6.0, -12.0, -18.0)):
        yaw_rad = math.radians(base_transform.rotation.yaw)
        fx = math.cos(yaw_rad)
        fy = math.sin(yaw_rad)
        for off in offsets_m:
            t = carla.Transform(
                carla.Location(
                    x=base_transform.location.x + fx * off,
                    y=base_transform.location.y + fy * off,
                    z=base_transform.location.z + 0.2,
                ),
                base_transform.rotation,
            )
            ator = self.world.try_spawn_actor(bp, t)
            if ator is not None:
                return ator
        return None

    def _extrair_estado_da_imagem(self):
        veiculos_ativos = [v for v in self.lista_atores if v is not None and v.is_alive and v.type_id.startswith('vehicle')]

        fila_via_ambulancia, fila_via_oposta, fila_via_perpendicular = 0, 0, 0
        tem_amb, conf_amb, dist_amb = 0.0, 0.0, 150.0

        ids_amb_ativos = {v.id for v in veiculos_ativos if 'ambulance' in v.type_id}
        self.ambulancias_ja_passaram.intersection_update(ids_amb_ativos)

        for v in veiculos_ativos:
            loc = v.get_location()
            via = self._classificar_via_veiculo(loc)
            vel = np.sqrt(v.get_velocity().x ** 2 + v.get_velocity().y ** 2)

            if 'ambulance' not in v.type_id and vel < self.LIMIAR_VELOCIDADE_FILA:
                if via == self.VIA_AMBULANCIA:
                    fila_via_ambulancia += 1
                elif via == self.VIA_OPOSTA:
                    fila_via_oposta += 1
                elif via == self.VIA_PERPENDICULAR:
                    fila_via_perpendicular += 1

            if 'ambulance' in v.type_id:
                _, s_amb, _ = self._metrica_para_trilha(loc, self.trilhas_vias[self.VIA_AMBULANCIA])
                s_cruz = self.s_cruzamento_por_via[self.VIA_AMBULANCIA]
                if s_amb >= (s_cruz + self.BUFFER_DESCARTE_AMB):
                    self.ambulancias_ja_passaram.add(v.id)
                if v.id not in self.ambulancias_ja_passaram and self._ambulancia_visivel_na_camera(loc):
                    tem_amb = 1.0
                    conf_amb = 1.0
                    dist_amb = loc.distance(self.posicao_cruzamento)

        self.espera_via_ambulancia = np.clip(
            self.espera_via_ambulancia + 1.0 if fila_via_ambulancia > 0 else max(0.0, self.espera_via_ambulancia - 0.5),
            0.0,
            100.0,
        )
        self.espera_via_oposta = np.clip(
            self.espera_via_oposta + 1.0 if fila_via_oposta > 0 else max(0.0, self.espera_via_oposta - 0.5),
            0.0,
            100.0,
        )
        self.espera_via_perpendicular = np.clip(
            self.espera_via_perpendicular + 1.0 if fila_via_perpendicular > 0 else max(0.0, self.espera_via_perpendicular - 0.5),
            0.0,
            100.0,
        )

        # Mantem a ordem do vetor de observacao para compatibilidade com o modelo.
        return np.array([fila_via_ambulancia, fila_via_oposta, fila_via_perpendicular,
                        self.espera_via_ambulancia, self.espera_via_oposta, self.espera_via_perpendicular,
                        tem_amb, dist_amb, conf_amb], dtype=np.float32)

    def _gerenciar_transito(self):
        veiculos_ativos = [v for v in self.lista_atores if v is not None and v.is_alive]
        ambulancias_ativas = [v for v in veiculos_ativos if 'ambulance' in v.type_id and v.id not in self.ambulancias_ja_passaram]

        # Gatilho de emergência (ex.: via 34 com fila ou pulsos temporais)
        fila_via_ambulancia_gt = sum(
            1 for v in veiculos_ativos
            if 'ambulance' not in v.type_id
            and self._classificar_via_veiculo(v.get_location()) == self.VIA_AMBULANCIA
            and np.sqrt(v.get_velocity().x ** 2 + v.get_velocity().y ** 2) < self.LIMIAR_VELOCIDADE_FILA
        )

        if len(ambulancias_ativas) == 0 and not self.ambulancia_pendente:
            if fila_via_ambulancia_gt >= 1 or (self.contador_frames % 300 == 0 and self.contador_frames > 100):
                self.ambulancia_pendente = True

        if self.ambulancia_pendente:
            amb = self._try_spawn_with_offsets(self.bp_ambulancia, self.pontos_spawn[self.VIA_AMBULANCIA])
            if amb is not None:
                amb.set_autopilot(True, self.tm.get_port())
                self.tm.set_desired_speed(amb, self.VELOCIDADE_AMB_FECHADO)
                self.tm.ignore_lights_percentage(amb, 0.0)
                self.tm.auto_lane_change(amb, False)
                amb.set_light_state(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Special1)
                self.lista_atores.append(amb)
                self.ambulancia_pendente = False

        # PULSOS & GOTEJAMENTO
        if self.contador_frames >= self.proximo_pulse:
            via_alvo = random.choices([self.VIA_AMBULANCIA, self.VIA_OPOSTA, self.VIA_PERPENDICULAR], weights=[0.6, 0.2, 0.2], k=1)[0]
            for _ in range(6):
                self.fila_de_pulsos.append(via_alvo)
            self.proximo_pulse = self.contador_frames + random.randint(150, 250)

        if len(veiculos_ativos) < self.max_veiculos_simultaneos:
            if self.fila_de_pulsos and self.contador_frames % 10 == 0:
                id_sorteado = self.fila_de_pulsos.pop(0)
                if self.ambulancia_pendente and id_sorteado == self.VIA_AMBULANCIA:
                    id_sorteado = random.choice([self.VIA_OPOSTA, self.VIA_PERPENDICULAR])
                carro = self._try_spawn_with_offsets(random.choice(self.bp_carros), self.pontos_spawn[id_sorteado])
                if carro is not None:
                    carro.set_autopilot(True, self.tm.get_port())
                    self.tm.ignore_lights_percentage(carro, 0.0)
                    self.tm.set_desired_speed(carro, self.VELOCIDADE_BASE_FECHADO)
                    self.lista_atores.append(carro)

            elif self.contador_frames % 25 == 0 and not self.fila_de_pulsos:
                id_sorteado = random.choice([self.VIA_AMBULANCIA, self.VIA_OPOSTA, self.VIA_PERPENDICULAR])
                if self.ambulancia_pendente and id_sorteado == self.VIA_AMBULANCIA:
                    id_sorteado = random.choice([self.VIA_OPOSTA, self.VIA_PERPENDICULAR])
                carro = self._try_spawn_with_offsets(random.choice(self.bp_carros), self.pontos_spawn[id_sorteado])
                if carro is not None:
                    carro.set_autopilot(True, self.tm.get_port())
                    self.tm.ignore_lights_percentage(carro, 0.0)
                    self.tm.set_desired_speed(carro, self.VELOCIDADE_BASE_FECHADO)
                    self.lista_atores.append(carro)

        if self.contador_frames % self.FRAMES_ATUALIZACAO_VELOCIDADE == 0:
            via_aberta = self.via_por_acao[self.acao_atual]
            for ator in veiculos_ativos:
                try:
                    if 'ambulance' in ator.type_id:
                        v_alvo = self.VELOCIDADE_AMB_ABERTO if via_aberta == self.VIA_AMBULANCIA else self.VELOCIDADE_AMB_FECHADO
                    else:
                        via_ator = self._classificar_via_veiculo(ator.get_location())
                        v_alvo = self.VELOCIDADE_BASE_ABERTO if via_ator == via_aberta else self.VELOCIDADE_BASE_FECHADO
                    self.tm.set_desired_speed(ator, v_alvo)
                except Exception:
                    pass

        for ator in veiculos_ativos:
            try:
                loc = ator.get_location()
                if loc.distance(self.posicao_cruzamento) > self.raio_despawn or loc.z < -10.0:
                    ator.destroy()
                    self.lista_atores.remove(ator)
            except Exception:
                if ator in self.lista_atores:
                    self.lista_atores.remove(ator)

        self.contador_frames += 1

    def step(self, action):
        self.ciclos_atuais += 1
        self.acao_atual = int(action)

        for i in range(3):
            if self.semaforos[i]:
                self.semaforos[i].set_state(carla.TrafficLightState.Red)
        if self.semaforos[action]:
            self.semaforos[action].set_state(carla.TrafficLightState.Green)

        custo_troca = 5.0 if (self.ultima_acao != -1 and action != self.ultima_acao) else 0.0
        self.ultima_acao = action

        for _ in range(self.ticks_por_step):
            self._gerenciar_transito()
            self.world.tick()

        novo_estado = self._extrair_estado_da_imagem()

        reward = self._calcular_recompensa(novo_estado, action) - custo_troca

        if self.ciclos_atuais % 10 == 0:
            fila_via_ambulancia, fila_via_oposta, fila_via_perpendicular = novo_estado[0], novo_estado[1], novo_estado[2]
            acoes = ["Via 34(Amb)", "Via 152", "Via 92"]
            print(
                f"⚙️ Step {self.ciclos_atuais:04d} | Ação: {acoes[action]:<12} | "
                f"Filas: amb={fila_via_ambulancia:.0f} oposta={fila_via_oposta:.0f} perpendicular={fila_via_perpendicular:.0f} | "
                f"Reward: {reward:+06.2f}"
            )

        terminated = False
        truncated = bool(self.ciclos_atuais >= self.max_ciclos)
        self.estado_atual = novo_estado
        return novo_estado, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ciclos_atuais = 0
        self.espera_via_ambulancia = 0.0
        self.espera_via_oposta = 0.0
        self.espera_via_perpendicular = 0.0
        self.contador_frames = 0
        self.ambulancia_pendente = False
        self.ambulancias_ja_passaram.clear()
        self.ultima_acao = -1
        self.acao_atual = 0

        for ator in self.lista_atores[:]:
            if ator.type_id.startswith('vehicle'):
                try:
                    if ator.is_alive:
                        ator.destroy()
                except Exception:
                    pass
        self.lista_atores = []

        for _ in range(40):
            self._gerenciar_transito()
            self.world.tick()

        self.estado_atual = self._extrair_estado_da_imagem()

        print("\n" + "="*50)
        print("🔄 NOVO EPISÓDIO INICIADO")
        print("="*50)

        return self.estado_atual, {}

    def _calcular_recompensa(self, estado, acao):
        (
            fila_via_ambulancia,
            fila_via_oposta,
            fila_via_perpendicular,
            espera_via_ambulancia,
            espera_via_oposta,
            espera_via_perpendicular,
            tem_amb,
            dist_amb,
            conf_amb,
        ) = estado
        reward = 0.0
        
        # Penalidade suavizada pelo tráfego normal
        total_filas = fila_via_ambulancia + fila_via_oposta + fila_via_perpendicular
        reward -= total_filas * 0.5  
        
        # Punição severa se a ambulância estiver presente e a via 34 NÃO estiver aberta
        if tem_amb == 1.0:
            if acao != 0:
                fator_distancia = 150.0 / (dist_amb + 1.0)
                reward -= (20.0 + fator_distancia)
            else:
                reward += 30.0
        else:
            max_espera = max(espera_via_ambulancia, espera_via_oposta, espera_via_perpendicular)
            if max_espera > 30.0:
                reward -= max_espera * 0.2

        return float(np.clip(reward, -100.0, 100.0))

    def close(self):
        print("\n🛑 Encerrando: Limpando atores e restaurando mundo...")
        for ator in self.lista_atores[:]:
            if ator is None:
                continue
            try:
                if ator.is_alive:
                    ator.destroy()
            except RuntimeError:
                pass

        try:
            for v in self.world.get_actors().filter('vehicle.*'):
                try:
                    v.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
        except Exception:
            pass
        print("✅ Mundo restaurado.")