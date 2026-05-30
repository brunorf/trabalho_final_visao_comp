import carla
import numpy as np
import time
import random
import math
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym

# ==============================================================================
# CLASSE AUXILIAR PARA CARREGAR O NORMALIZADOR SEM PRECISAR DO ARQUIVO DE TREINO
# ==============================================================================
class DummyEnvSimulada(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,))
        self.action_space = gym.spaces.Discrete(3)

# ==============================================================================
# FUNÇÃO PRINCIPAL DA DEMONSTRAÇÃO
# ==============================================================================
def main():
    print("🎬 Inicializando Sistema Autônomo Completo (Gráficos + Tráfego + IA)...")

    ARQUIVO_CALIBRACAO_SPAWN = "spawn_refs.json"
    
    # 1. CONEXÃO COM CARLA
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    print(f"[INFO] Mapa carregado manualmente no CARLA: {world.get_map().name}")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False # GRAFICOS LIGADOS PARA O SEU VÍDEO!
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(1.5)

    # 2. CONFIGURAÇÕES DA INFRAESTRUTURA E ATORES
    posicao_cruzamento = carla.Location(x=323, y=320, z=3)
    area_cruzamento = {'x_min': 310, 'x_max': 335, 'y_min': 310, 'y_max': 335}
    
    blueprint_library = world.get_blueprint_library()
    
    bp_carros = [
        blueprint_library.find('vehicle.audi.a2'), blueprint_library.find('vehicle.audi.etron'),
        blueprint_library.find('vehicle.audi.tt'), blueprint_library.find('vehicle.bmw.grandtourer'),
        blueprint_library.find('vehicle.chevrolet.impala'), blueprint_library.find('vehicle.citroen.c3'),
        blueprint_library.find('vehicle.ford.crown'), blueprint_library.find('vehicle.ford.mustang'),
        blueprint_library.find('vehicle.mercedes.coupe'), blueprint_library.find('vehicle.micro.microlino'),
        blueprint_library.find('vehicle.nissan.micra'), blueprint_library.find('vehicle.nissan.patrol'),
        blueprint_library.find('vehicle.seat.leon'), blueprint_library.find('vehicle.tesla.model3'),
        blueprint_library.find('vehicle.toyota.prius'), blueprint_library.find('vehicle.volkswagen.t2'),
        blueprint_library.find('vehicle.kawasaki.ninja'), blueprint_library.find('vehicle.vespa.zx125'),
        blueprint_library.find('vehicle.yamaha.yzf'), blueprint_library.find('vehicle.bh.crossbike'),
        blueprint_library.find('vehicle.diamondback.century'), blueprint_library.find('vehicle.gazelle.omafiets')
    ]
    bp_ambulancia = blueprint_library.find('vehicle.ford.ambulance')
    
    spawn_points = world.get_map().get_spawn_points()
    VIA_AMBULANCIA = "via_ambulancia"
    VIA_OPOSTA = "via_oposta"
    VIA_PERPENDICULAR = "via_perpendicular"
    vias = [VIA_AMBULANCIA, VIA_OPOSTA, VIA_PERPENDICULAR]

    ACAO_VIA_AMBULANCIA = 0
    ACAO_VIA_OPOSTA = 1
    ACAO_VIA_PERPENDICULAR = 2

    legacy_spawn_index_por_via = {
        VIA_AMBULANCIA: 34,
        VIA_OPOSTA: 152,
        VIA_PERPENDICULAR: 92,
    }

    # Modo recomendado: travar os spawns por coordenada de referencia no mapa.
    # Isso evita dependencia da ordem de spawn_points, que pode mudar entre versoes/mapas.
    USAR_SPAWN_POR_REFERENCIA = True
    REFERENCIAS_SPAWN = {
        # Via da ambulancia (aprox. ao sul do cruzamento, seguindo para o cruzamento)
        VIA_AMBULANCIA: carla.Location(x=322.5, y=280.0, z=0.5),
        # Via oposta (aprox. ao norte do cruzamento)
        VIA_OPOSTA: carla.Location(x=322.5, y=365.0, z=0.5),
        # Via perpendicular (aprox. a oeste do cruzamento)
        VIA_PERPENDICULAR: carla.Location(x=280.0, y=322.5, z=0.5),
    }

    def location_to_dict(loc):
        return {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)}

    def dict_to_location(d):
        return carla.Location(x=float(d["x"]), y=float(d["y"]), z=float(d.get("z", 0.5)))

    def carregar_calibracao(path_arquivo):
        if not os.path.exists(path_arquivo):
            return None
        try:
            with open(path_arquivo, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapa_calibrado = data.get("map_name")
            refs = data.get("references", {})
            # Formato novo (semântico)
            if all(via in refs for via in vias):
                out = {
                    VIA_AMBULANCIA: dict_to_location(refs[VIA_AMBULANCIA]),
                    VIA_OPOSTA: dict_to_location(refs[VIA_OPOSTA]),
                    VIA_PERPENDICULAR: dict_to_location(refs[VIA_PERPENDICULAR]),
                }
            else:
                # Formato antigo (34/152/92)
                legacy_keys = [
                    str(legacy_spawn_index_por_via[VIA_AMBULANCIA]),
                    str(legacy_spawn_index_por_via[VIA_OPOSTA]),
                    str(legacy_spawn_index_por_via[VIA_PERPENDICULAR]),
                ]
                if not all(k in refs for k in legacy_keys):
                    return None
                out = {
                    VIA_AMBULANCIA: dict_to_location(refs[str(legacy_spawn_index_por_via[VIA_AMBULANCIA])]),
                    VIA_OPOSTA: dict_to_location(refs[str(legacy_spawn_index_por_via[VIA_OPOSTA])]),
                    VIA_PERPENDICULAR: dict_to_location(refs[str(legacy_spawn_index_por_via[VIA_PERPENDICULAR])]),
                }
            if mapa_calibrado and mapa_calibrado != world.get_map().name:
                print(
                    f"[CALIB] Arquivo de calibracao foi criado em '{mapa_calibrado}', "
                    f"mas o mapa atual e '{world.get_map().name}'."
                )
            return out
        except Exception as e:
            print(f"[CALIB] Falha ao carregar calibracao '{path_arquivo}': {e}")
            return None

    calibracao_arquivo = carregar_calibracao(ARQUIVO_CALIBRACAO_SPAWN)
    if calibracao_arquivo is not None:
        REFERENCIAS_SPAWN = calibracao_arquivo
        print(f"[CALIB] Referencias carregadas de '{ARQUIVO_CALIBRACAO_SPAWN}'.")
    else:
        print(
            f"[CALIB] Arquivo '{ARQUIVO_CALIBRACAO_SPAWN}' nao encontrado. "
            "Usando referencias padrao embutidas no script."
        )

    ids_spawns = vias

    def spawn_transform_from_reference(ref_loc, metros_recuo=14.0):
        mapa = world.get_map()
        wp = mapa.get_waypoint(ref_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            raise RuntimeError(f"Nao foi possivel projetar referencia de spawn na via: {ref_loc}")

        # Recuamos no sentido contrario da via para abrir espaco e reduzir bloqueio no spawn.
        passos = int(metros_recuo // 2)
        for _ in range(max(passos, 0)):
            prev = wp.previous(2.0)
            if not prev:
                break
            wp = prev[0]

        t = wp.transform
        t.location.z += 0.3
        return t

    if USAR_SPAWN_POR_REFERENCIA:
        pontos_spawn = {
            i: spawn_transform_from_reference(REFERENCIAS_SPAWN[i]) for i in ids_spawns
        }
        print(
            "[SPAWN] Modo referencia ativo (coordenadas fixas), independente da ordem de spawn_points."
        )
    else:
        if max(legacy_spawn_index_por_via.values()) >= len(spawn_points):
            raise RuntimeError(
                f"IDs de spawn inválidos para o mapa atual. Máximo pedido={max(legacy_spawn_index_por_via.values())}, disponíveis={len(spawn_points)-1}"
            )
        pontos_spawn = {
            VIA_AMBULANCIA: spawn_points[legacy_spawn_index_por_via[VIA_AMBULANCIA]],
            VIA_OPOSTA: spawn_points[legacy_spawn_index_por_via[VIA_OPOSTA]],
            VIA_PERPENDICULAR: spawn_points[legacy_spawn_index_por_via[VIA_PERPENDICULAR]],
        }
        print("[SPAWN] Modo ID ativo (dependente da ordem de spawn_points).")

    # Contagem de filas por corredor dinamico de aproximacao (suporta vias com curva).
    CORREDOR_LARGURA_MEIA = 6.0
    MARGEM_CRUZAMENTO = 8.0
    PASSO_WAYPOINT_METROS = 2.0
    MAX_PONTOS_TRILHA = 180
    PASSOS_EXTRA_APOS_CRUZAMENTO = 12
    BUFFER_FIM_VISAO_CAMERA_AMB = 3.0
    BUFFER_DESCARTE_AMB = 12.0

    def dist2d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def distancia_ponto_segmento_2d(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        if denom < 1e-9:
            # Segmento degenerado
            t = 0.0
            qx, qy = ax, ay
        else:
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
            qx = ax + t * abx
            qy = ay + t * aby
        d = math.hypot(px - qx, py - qy)
        return d, t

    def construir_trilha_ate_cruzamento(spawn_transform):
        """Segue waypoints da via desde o spawn ate perto do cruzamento (inclui curvas)."""
        mapa = world.get_map()
        wp = mapa.get_waypoint(
            spawn_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return [spawn_transform.location], set()

        trilha = [wp.transform.location]
        chaves_via = {(wp.road_id, wp.lane_id)}
        visitados = {(wp.road_id, wp.lane_id, round(wp.s, 1))}

        chegou_cruzamento = False
        passos_pos_cruzamento = 0

        for _ in range(MAX_PONTOS_TRILHA):
            if dist2d(wp.transform.location, posicao_cruzamento) <= MARGEM_CRUZAMENTO:
                chegou_cruzamento = True

            if chegou_cruzamento and passos_pos_cruzamento >= PASSOS_EXTRA_APOS_CRUZAMENTO:
                break

            candidatos = wp.next(PASSO_WAYPOINT_METROS)
            if not candidatos:
                break

            # Escolhe o proximo waypoint que mais aproxima do cruzamento.
            prox = min(candidatos, key=lambda c: dist2d(c.transform.location, posicao_cruzamento))
            chave = (prox.road_id, prox.lane_id, round(prox.s, 1))
            if chave in visitados:
                break

            visitados.add(chave)
            wp = prox
            trilha.append(wp.transform.location)
            chaves_via.add((wp.road_id, wp.lane_id))
            if chegou_cruzamento:
                passos_pos_cruzamento += 1

        if len(trilha) == 1:
            # fallback: corredor minimo entre origem e cruzamento
            trilha.append(posicao_cruzamento)
        return trilha, chaves_via

    trilhas_vias = {}
    chaves_lane_por_via = {}
    for via_id in ids_spawns:
        trilha, chaves_via = construir_trilha_ate_cruzamento(pontos_spawn[via_id])
        trilhas_vias[via_id] = trilha
        chaves_lane_por_via[via_id] = chaves_via

    def metrica_para_trilha(loc, trilha):
        """Retorna (distancia lateral minima, progresso_s, comprimento_total)."""
        if len(trilha) < 2:
            d = dist2d(loc, trilha[0]) if trilha else float('inf')
            return d, 0.0, 0.0

        px, py = loc.x, loc.y
        melhor_d = float('inf')
        melhor_s = 0.0
        s_acum = 0.0
        comprimento_total = 0.0

        for i in range(len(trilha) - 1):
            a = trilha[i]
            b = trilha[i + 1]
            seg_len = dist2d(a, b)
            d, t = distancia_ponto_segmento_2d(px, py, a.x, a.y, b.x, b.y)
            if d < melhor_d:
                melhor_d = d
                melhor_s = s_acum + t * seg_len
            s_acum += seg_len

        comprimento_total = s_acum
        return melhor_d, melhor_s, comprimento_total

    s_cruzamento_por_via = {}
    for via_id in ids_spawns:
        _, s_cruz, _ = metrica_para_trilha(posicao_cruzamento, trilhas_vias[via_id])
        s_cruzamento_por_via[via_id] = s_cruz

    def esta_no_corredor_da_via(loc, via_id):
        trilha = trilhas_vias[via_id]
        lateral, progresso_s, comp_total = metrica_para_trilha(loc, trilha)

        # Faixa da trilha + margem para incluir a boca do cruzamento.
        if progresso_s < 0.0 or progresso_s > (comp_total + MARGEM_CRUZAMENTO):
            return False
        return lateral <= CORREDOR_LARGURA_MEIA

    def classificar_via_veiculo(loc):
        """Classifica um veiculo em uma via usando lane waypoint + corredor da trilha."""
        wp_v = world.get_map().get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp_v is None:
            return None

        chave_v = (wp_v.road_id, wp_v.lane_id)
        candidatos = [
            via_id for via_id in ids_spawns if chave_v in chaves_lane_por_via[via_id]
        ]

        if not candidatos:
            return None

        # Se houver ambiguidade, escolhe o corredor com menor distancia lateral.
        melhor_via = None
        melhor_lateral = float("inf")
        for via_id in candidatos:
            if not esta_no_corredor_da_via(loc, via_id):
                continue
            lateral, _, _ = metrica_para_trilha(loc, trilhas_vias[via_id])
            if lateral < melhor_lateral:
                melhor_lateral = lateral
                melhor_via = via_id
        return melhor_via

    def ambulancia_visivel_na_camera(loc):
        """Retorna True somente enquanto a ambulancia esta no campo de visao da camera da via 34.

        Regras:
        - precisa estar no corredor da via da ambulancia;
        - precisa estar antes da linha do cruzamento/semaforo (aproximacao).
        """
        trilha_amb = trilhas_vias[VIA_AMBULANCIA]
        lateral, progresso_s, comp_total = metrica_para_trilha(loc, trilha_amb)
        s_cruz = s_cruzamento_por_via[VIA_AMBULANCIA]

        if lateral > CORREDOR_LARGURA_MEIA:
            return False
        if progresso_s < 0.0:
            return False

        # Mantem detectada ate alguns metros apos a linha do semaforo/cruzamento.
        limite_saida_camera = min(comp_total, s_cruz + BUFFER_FIM_VISAO_CAMERA_AMB)
        return progresso_s <= limite_saida_camera

    def try_spawn_with_offsets(bp, base_transform, offsets_m=(0.0, -6.0, -12.0, -18.0)):
        """Tenta spawn no ponto base e alguns metros para tras para escapar de bloqueio local."""
        yaw_rad = math.radians(base_transform.rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)

        for offset in offsets_m:
            t = carla.Transform(
                carla.Location(
                    x=base_transform.location.x + forward_x * offset,
                    y=base_transform.location.y + forward_y * offset,
                    z=base_transform.location.z + 0.2,
                ),
                base_transform.rotation,
            )
            ator = world.try_spawn_actor(bp, t)
            if ator is not None:
                return ator
        return None

    def nearest_vehicle_distance_to_spawn(base_transform, ativos):
        dmin = float('inf')
        for a in ativos:
            try:
                if a is not None and a.is_alive and a.type_id.startswith('vehicle'):
                    d = a.get_location().distance(base_transform.location)
                    if d < dmin:
                        dmin = d
            except Exception:
                continue
        return dmin if dmin != float('inf') else None

    def actor_is_alive_safe(actor):
        try:
            return actor is not None and actor.is_alive
        except Exception:
            return False

    def actor_location_safe(actor):
        try:
            return actor.get_location()
        except Exception:
            return None

    distancias_spawns = [
        pontos_spawn[i].location.distance(posicao_cruzamento) for i in ids_spawns
    ]
    raio_despawn = max(distancias_spawns) + 20.0
    print(
        f"[SPAWN] Distâncias ao cruzamento (34/152/92): "
        f"{distancias_spawns[0]:.1f}m / {distancias_spawns[1]:.1f}m / {distancias_spawns[2]:.1f}m | "
        f"Raio despawn: {raio_despawn:.1f}m"
    )
    
    lista_atores = []

    todos_semaforos = [
        tl for tl in world.get_actors().filter('traffic.traffic_light')
        if tl.get_location().distance(posicao_cruzamento) < 35.0
    ]

    def mapear_semaforos_por_via(semaforos):
        restantes = semaforos[:]
        mapeados = {}
        for via_id in ids_spawns:
            if not restantes:
                break
            origem = pontos_spawn[via_id].location
            escolhido = min(restantes, key=lambda s: s.get_location().distance(origem))
            mapeados[via_id] = escolhido
            restantes.remove(escolhido)
        return mapeados

    semaforos_por_via = mapear_semaforos_por_via(todos_semaforos)
    lista_semaforos_ordenada = [
        semaforos_por_via[VIA_AMBULANCIA],
        semaforos_por_via[VIA_OPOSTA],
        semaforos_por_via[VIA_PERPENDICULAR],
    ] if len(semaforos_por_via) == 3 else todos_semaforos[:3]

    for tl in lista_semaforos_ordenada:
        tl.freeze(True)
        tl.set_state(carla.TrafficLightState.Red)

    print(
        "[SEMAFORO] IDs controlados (amb/oposta/perpendicular): "
        + ", ".join(str(tl.id) for tl in lista_semaforos_ordenada)
    )

    # 3. CARREGANDO O CÉREBRO DA IA (Altere o caminho para onde os arquivos estiverem na máquina nova)
    print("🧠 Carregando IA...")
    caminho_modelo = "./modelos/modelo_fase2_2" # Coloque o nome do arquivo .zip sem a extensão
    caminho_norm = "./modelos/vec_normalize_2.pkl"
    
    
    modelo = PPO.load(caminho_modelo, device='cpu')
    ambiente_falso = DummyVecEnv([lambda: DummyEnvSimulada()])
    vec_normalize = VecNormalize.load(caminho_norm, ambiente_falso)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    print("🟢 DEMONSTRAÇÃO INICIADA! (Pressione CTRL+C para encerrar)")

    VELOCIDADE_BASE_FECHADO = 4.0
    VELOCIDADE_BASE_ABERTO = 9.0
    VELOCIDADE_AMB_FECHADO = 5.0
    VELOCIDADE_AMB_ABERTO = 13.0
    FRAMES_ATUALIZACAO_VELOCIDADE = 10
    
    contador_frames = 0
    acao_atual = 0
    ambulancia_pendente = False
    falhas_spawn = {VIA_AMBULANCIA: 0, VIA_OPOSTA: 0, VIA_PERPENDICULAR: 0}
    via_por_acao = [VIA_AMBULANCIA, VIA_OPOSTA, VIA_PERPENDICULAR]
    ambulancias_ja_passaram = set()
    
    try:
        while True:
            # ==============================================================================
            # MOTOR DE TRÁFEGO (Spawn, Despawn e Ambulância)
            # ==============================================================================
            # Limpa handles inválidos para evitar chamada em atores já destruídos no backend do CARLA.
            veiculos_ativos = []
            for v in lista_atores[:]:
                if actor_is_alive_safe(v):
                    veiculos_ativos.append(v)
                else:
                    if v in lista_atores:
                        lista_atores.remove(v)

            ambulancias_ativas = [v for v in veiculos_ativos if 'ambulance' in v.type_id]
            
            # Nasce ambulância a cada 400 frames se não houver nenhuma
            if len(ambulancias_ativas) == 0 and not ambulancia_pendente and contador_frames % 400 == 0 and contador_frames > 50:
                ambulancia_pendente = True

            if ambulancia_pendente:
                amb = try_spawn_with_offsets(bp_ambulancia, pontos_spawn[VIA_AMBULANCIA])
                if amb is not None:
                    amb.set_autopilot(True, tm.get_port())
                    tm.set_desired_speed(amb, 5.0)
                    tm.ignore_lights_percentage(amb, 0.0)
                    # Solução do Bug do C++ corrigida aqui também!
                    amb.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.Special1))
                    lista_atores.append(amb)
                    ambulancia_pendente = False
                    print(f"[SPAWN-AMB] Ambulância criada na {VIA_AMBULANCIA}.")
                elif contador_frames % 100 == 0:
                    dmin = nearest_vehicle_distance_to_spawn(pontos_spawn[VIA_AMBULANCIA], veiculos_ativos)
                    if dmin is None:
                        print(f"[SPAWN-AMB] Sem sucesso na {VIA_AMBULANCIA}; sem veículos próximos, possível bloqueio geométrico.")
                    else:
                        print(f"[SPAWN-AMB] Sem sucesso na {VIA_AMBULANCIA}; veículo mais próximo a {dmin:.1f}m.")

            # Gotejamento de trânsito normal
            if len(veiculos_ativos) < 35 and contador_frames % 25 == 0:
                ids_candidatos = [VIA_AMBULANCIA, VIA_OPOSTA, VIA_PERPENDICULAR]
                if ambulancia_pendente:
                    # Reserva o corredor da ambulancia enquanto ela nao consegue spawnar.
                    ids_candidatos = [VIA_OPOSTA, VIA_PERPENDICULAR]

                random.shuffle(ids_candidatos)
                spawned = False
                carro_bp = random.choice(bp_carros)

                for id_sorteado in ids_candidatos:
                    carro = try_spawn_with_offsets(carro_bp, pontos_spawn[id_sorteado])
                    if carro is not None:
                        carro.set_autopilot(True, tm.get_port())
                        tm.set_desired_speed(carro, 4.0)
                        lista_atores.append(carro)
                        spawned = True
                        break
                    falhas_spawn[id_sorteado] += 1

                if not spawned and contador_frames % 100 == 0:
                    d34 = nearest_vehicle_distance_to_spawn(pontos_spawn[VIA_AMBULANCIA], veiculos_ativos)
                    d152 = nearest_vehicle_distance_to_spawn(pontos_spawn[VIA_OPOSTA], veiculos_ativos)
                    d92 = nearest_vehicle_distance_to_spawn(pontos_spawn[VIA_PERPENDICULAR], veiculos_ativos)
                    print(
                        f"[SPAWN-FALHA] amb={falhas_spawn[VIA_AMBULANCIA]} "
                        f"oposta={falhas_spawn[VIA_OPOSTA]} perpendicular={falhas_spawn[VIA_PERPENDICULAR]} | "
                        f"dist_min: amb={d34 if d34 is not None else 'None'} "
                        f"oposta={d152 if d152 is not None else 'None'} "
                        f"perpendicular={d92 if d92 is not None else 'None'}"
                    )

            # Limpeza de carros (Destruição natural)
            for ator in veiculos_ativos:
                try:
                    if not actor_is_alive_safe(ator):
                        if ator in lista_atores:
                            lista_atores.remove(ator)
                        continue

                    loc = actor_location_safe(ator)
                    if loc is None:
                        if ator in lista_atores:
                            lista_atores.remove(ator)
                        continue

                    if loc.distance(posicao_cruzamento) > raio_despawn or loc.z < -10.0:
                        ator.destroy()
                        if ator in lista_atores:
                            lista_atores.remove(ator)
                except Exception:
                    if ator in lista_atores:
                        lista_atores.remove(ator)

            # Acelera veiculos da via atualmente aberta e segura os demais em velocidade base.
            if contador_frames % FRAMES_ATUALIZACAO_VELOCIDADE == 0:
                via_aberta = via_por_acao[acao_atual]
                for ator in veiculos_ativos:
                    try:
                        loc_ator = ator.get_location()
                        via_ator = classificar_via_veiculo(loc_ator)

                        if 'ambulance' in ator.type_id:
                            # Ambulancia responde mais forte ao verde da propria via.
                            velocidade_alvo = VELOCIDADE_AMB_FECHADO
                            if via_aberta == VIA_AMBULANCIA:
                                velocidade_alvo = VELOCIDADE_AMB_ABERTO
                        else:
                            velocidade_alvo = VELOCIDADE_BASE_FECHADO
                            if via_ator == via_aberta:
                                velocidade_alvo = VELOCIDADE_BASE_ABERTO

                        tm.set_desired_speed(ator, velocidade_alvo)
                    except:
                        pass

            # ==============================================================================
            # MOTOR DE INFERÊNCIA RL (Decisão do Semáforo a cada 10 segundos / 200 frames)
            # ==============================================================================
            if contador_frames % 200 == 0:
                fila_34, fila_152, fila_92 = 0, 0, 0
                tem_amb, conf_amb, dist_amb = 0.0, 0.0, 150.0 

                # Limpa IDs de ambulancias que ja foram destruidas.
                ids_ambulancias_ativas = {v.id for v in veiculos_ativos if 'ambulance' in v.type_id}
                ambulancias_ja_passaram.intersection_update(ids_ambulancias_ativas)
                
                # Leitura Ground Truth pela API do CARLA, usando corredores dinamicos por via.
                for v in veiculos_ativos:
                    if not actor_is_alive_safe(v):
                        continue

                    loc = actor_location_safe(v)
                    if loc is None:
                        continue

                    dist = loc.distance(posicao_cruzamento)

                    via_classificada = classificar_via_veiculo(loc)
                    if via_classificada == VIA_AMBULANCIA:
                        fila_34 += 1
                    elif via_classificada == VIA_OPOSTA:
                        fila_152 += 1
                    elif via_classificada == VIA_PERPENDICULAR:
                        fila_92 += 1

                    if 'ambulance' in v.type_id:
                        # So marca como "passou" apos ultrapassar o cruzamento com folga.
                        _, progresso_amb, _ = metrica_para_trilha(loc, trilhas_vias[VIA_AMBULANCIA])
                        s_cruz_amb = s_cruzamento_por_via[VIA_AMBULANCIA]
                        if progresso_amb >= (s_cruz_amb + BUFFER_DESCARTE_AMB):
                            ambulancias_ja_passaram.add(v.id)

                        if v.id in ambulancias_ja_passaram:
                            continue

                        if ambulancia_visivel_na_camera(loc):
                            tem_amb, conf_amb, dist_amb = 1.0, 1.0, dist

                estado_cru = np.array([fila_34, fila_152, fila_92, 0.0, 0.0, 0.0, tem_amb, dist_amb, conf_amb], dtype=np.float32)
                
                # Pede para a Rede Neural decidir
                estado_normalizado = vec_normalize.normalize_obs(estado_cru)
                acao_atual, _ = modelo.predict(estado_normalizado, deterministic=True)
                
                vias = ["Via 34(Amb)", "Via 152", "Via 92"]
                print(f"🚦 Decisão: {vias[acao_atual]} | Filas: 34={fila_34} 152={fila_152} 92={fila_92} | Amb: {'Sim' if tem_amb else 'Não'}")

                # Aplica a cor aos faróis físicos
                for i, semaforo in enumerate(lista_semaforos_ordenada):
                    if i >= 3:
                        continue
                    try:
                        if not actor_is_alive_safe(semaforo):
                            continue
                        if i == acao_atual:
                            semaforo.set_state(carla.TrafficLightState.Green)
                        else:
                            semaforo.set_state(carla.TrafficLightState.Red)
                    except Exception:
                        continue

            world.tick()
            contador_frames += 1
            
            # Segura o tempo para não rodar rápido demais e estragar seu vídeo
            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\n⏹️ Demonstração finalizada. Limpando mapa...")
    finally:
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        for ator in lista_atores:
            if ator is not None and ator.is_alive:
                ator.destroy()

if __name__ == "__main__":
    main()