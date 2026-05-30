# 152, 92, 34
import carla
import random
import cv2
import numpy as np
import sys

# ==============================================================================
# FUNÇÕES DE CONFIGURAÇÃO E INFRAESTRUTURA
# ==============================================================================

def configurar_mundo():
    """Inicializa o CARLA, o Modo Síncrono e o Traffic Manager acelerado."""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        if not world.get_map().name.endswith('Town01'):
            print("Carregando Town01...")
            world = client.load_world('Town01')
            
        bp_lib = world.get_blueprint_library()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.08 
        world.apply_settings(settings)
        
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(1.5) 
        tm.set_random_device_seed(0)
        
        print("[SISTEMA] Mundo síncrono e Traffic Manager configurados.")
        return client, world, bp_lib, tm
    except Exception as e:
        print(f"Erro na configuração do mundo: {e}")
        sys.exit(1)

def processar_imagem_camera(imagem, frame_atual):
    """Converte o dado bruto da câmera para BGR (OpenCV)."""
    array = np.frombuffer(imagem.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (imagem.height, imagem.width, 4))
    array = array[:, :, :3]
    frame_atual[0] = array

def instanciar_camera_fixa(world, bp_lib):
    """Instancia a câmera de CFTV."""
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    
    transform_camera = carla.Transform(
        carla.Location(x=327.28, y=138.37, z=3.0), 
        carla.Rotation(pitch=0.0, yaw=45.0)
    )
    return world.spawn_actor(cam_bp, transform_camera)

def configurar_semaforos_globais(world, localizacao_central):
    """Isola o cruzamento alvo e força VERDE ETERNO no resto do mapa."""
    todos_semaforos = world.get_actors().filter('traffic.traffic_light')
    semaforos_do_cruzamento = []
    
    for semaforo in todos_semaforos:
        if semaforo.get_location().distance(localizacao_central) < 35.0:
            semaforo.freeze(True)
            semaforos_do_cruzamento.append(semaforo)
        else:
            semaforo.freeze(False) 
            semaforo.set_state(carla.TrafficLightState.Green)
            semaforo.freeze(True)  
            
    print(f"[SEMAFOROS] {len(semaforos_do_cruzamento)} postes manuais. Resto VERDE.")
    return semaforos_do_cruzamento


# ==============================================================================
# LOOP PRINCIPAL DA SIMULAÇÃO DE TRÁFEGO
# ==============================================================================

if __name__ == "__main__":
    client, world, bp_lib, tm = configurar_mundo()
    
    # 1. NOVA COORDENADA DO CRUZAMENTO (Centro de Referência)
    posicao_cruzamento = carla.Location(x=323.0, y=320.0, z=2.0)
    
    tm.set_global_distance_to_leading_vehicle(1.5)
    
    # NOVA CÂMERA DE CFTV
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    
    transform_camera = carla.Transform(
        carla.Location(x=323.0, y=320.0, z=3.0), 
        carla.Rotation(pitch=0.0, yaw=180.0)
    )
    camera = world.spawn_actor(cam_bp, transform_camera)
    
    frame_atual = [None]
    camera.listen(lambda dados: processar_imagem_camera(dados, frame_atual))
    
    lista_atores = [camera]
    semaforos_locais = configurar_semaforos_globais(world, posicao_cruzamento)
    
    # 2. NOVOS PONTOS DE SPAWN
    spawn_points = world.get_map().get_spawn_points()
    
    # --- A SUA NOVA CONFIGURAÇÃO EXATA ---
    ID_AMBULANCIA = 34      # Avenida reta (De frente para a sua câmera)
    ID_OPOSTO = 152         # Avenida com curva (Sentido contrário)
    ID_PERPENDICULAR = 92   # Via transversal
    
    ponto_spawn_ambulancia = spawn_points[ID_AMBULANCIA]     
    ponto_spawn_oposto = spawn_points[ID_OPOSTO]          
    ponto_spawn_perpendicular = spawn_points[ID_PERPENDICULAR]
    
    distancias_spawns = [
        ponto_spawn_ambulancia.location.distance(posicao_cruzamento),
        ponto_spawn_oposto.location.distance(posicao_cruzamento),
        ponto_spawn_perpendicular.location.distance(posicao_cruzamento)
    ]
    raio_despawn = max(distancias_spawns) + 20.0
    
    # 3. BLUEPRINTS DOS ATORES
    # bp_carros = (
        # list(bp_lib.filter('vehicle.audi.*')) + 
        # list(bp_lib.filter('vehicle.tesla.*')) + 
        # list(bp_lib.filter('vehicle.citroen.*'))
    # )
    
    bp_carros = [
        bp_lib.find('vehicle.audi.a2'),
        bp_lib.find('vehicle.audi.etron'),
        bp_lib.find('vehicle.audi.tt'),
        bp_lib.find('vehicle.bmw.grandtourer'),
        bp_lib.find('vehicle.chevrolet.impala'),
        bp_lib.find('vehicle.citroen.c3'),
        bp_lib.find('vehicle.ford.crown'),
        bp_lib.find('vehicle.ford.mustang'),
        bp_lib.find('vehicle.mercedes.coupe'),
        bp_lib.find('vehicle.micro.microlino'),
        bp_lib.find('vehicle.nissan.micra'),
        bp_lib.find('vehicle.nissan.patrol'),
        bp_lib.find('vehicle.seat.leon'),
        bp_lib.find('vehicle.tesla.model3'),
        bp_lib.find('vehicle.toyota.prius'),
        bp_lib.find('vehicle.ford.ambulance'),
        bp_lib.find('vehicle.volkswagen.t2'),
        # bp_lib.find('vehicle.mitsubishi.fusorosa'),
        bp_lib.find('vehicle.kawasaki.ninja'),
        bp_lib.find('vehicle.vespa.zx125'),
        bp_lib.find('vehicle.yamaha.yzf'),
        bp_lib.find('vehicle.bh.crossbike'),
        bp_lib.find('vehicle.diamondback.century'),
        bp_lib.find('vehicle.gazelle.omafiets')
    ]
    bp_ambulancia = bp_lib.find('vehicle.ford.ambulance')
    
    # 4. VARIÁVEIS DE CONTROLE DO CAOS
    contador_frames = 0
    frequencia_spawn_carros = 8
    frequencia_ambulancia = 800      
    max_veiculos_simultaneos = 120
    ambulancia_pendente = False      
    
    print(f"\n[INFO] Simulação Simplificada Iniciada no Novo Cruzamento. Raio de Despawn: {raio_despawn:.1f}m")
    try:
        while True:
            # --- FASE 1: SEMÁFORO INDEPENDENTE ---
            total_fases = len(semaforos_locais)
            if total_fases > 0:
                ciclo = (contador_frames // 400) % total_fases 
                for indice, semaforo in enumerate(semaforos_locais):
                    if indice == ciclo:
                        semaforo.set_state(carla.TrafficLightState.Green)
                    else:
                        semaforo.set_state(carla.TrafficLightState.Red)
            
            veiculos_ativos = [a for a in lista_atores if a.is_alive and a.type_id.startswith('vehicle')]
            ambulancias_ativas = [a for a in veiculos_ativos if 'ambulance' in a.type_id]
            
            # --- FASE 2: GATILHO DA AMBULÂNCIA ---
            if contador_frames > 0 and contador_frames % frequencia_ambulancia == 0:
                if len(ambulancias_ativas) == 0 and not ambulancia_pendente:
                    ambulancia_pendente = True
                    print(f"\n[SISTEMA] Emergência! Solicitando ambulância no ponto {ID_AMBULANCIA}.")

            if ambulancia_pendente:
                ambulancia = world.try_spawn_actor(bp_ambulancia, ponto_spawn_ambulancia)
                if ambulancia is not None:
                    ambulancia.set_simulate_physics(True)
                    luzes = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
                    ambulancia.set_light_state(carla.VehicleLightState(luzes))
                    
                    ambulancia.set_target_velocity(ponto_spawn_ambulancia.get_forward_vector() * 7.0)
                    ambulancia.set_autopilot(True, tm.get_port())
                    
                    tm.ignore_lights_percentage(ambulancia, 0.0)
                    tm.auto_lane_change(ambulancia, False)
                    tm.set_desired_speed(ambulancia, 12.0)
                    
                    # Garantimos que a ambulância cruze o semáforo sem desvios
                    tm.set_route(ambulancia, ["Straight", "Straight"]) 
                    
                    lista_atores.append(ambulancia)
                    ambulancia_pendente = False                   
                    print(f"[EMERGÊNCIA] Ambulância {ambulancia.id} ativa e marchando.\n")

            # --- FASE 3: SPAWN DE VEÍCULOS COMUNS (TOTALMENTE LIVRES) ---
            if contador_frames % frequencia_spawn_carros == 0 and len(veiculos_ativos) < max_veiculos_simultaneos:
                id_sorteado = random.choice([ID_AMBULANCIA, ID_OPOSTO, ID_PERPENDICULAR])
                if ambulancia_pendente and id_sorteado == ID_AMBULANCIA:
                    id_sorteado = random.choice([ID_OPOSTO, ID_PERPENDICULAR])
                    
                ponto_origem = spawn_points[id_sorteado]
                carro_bp = random.choice(bp_carros)
                
                carro = world.try_spawn_actor(carro_bp, ponto_origem)
                if carro is not None:
                    carro.set_simulate_physics(True)
                    luzes = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
                    carro.set_light_state(carla.VehicleLightState(luzes))
                    carro.set_target_velocity(ponto_origem.get_forward_vector() * 5.0)
                    carro.set_autopilot(True, tm.get_port())
                    
                    tm.ignore_lights_percentage(carro, 0.0)
                    tm.auto_lane_change(carro, False)
                    tm.set_desired_speed(carro, 12.0) 
                    
                    # SEM NENHUM COMANDO DE ROTA! A IA do CARLA decide tudo.
                    
                    lista_atores.append(carro)

            # --- FASE 4: LIMPEZA DE DESPAWN DINÂMICO BLINDADA ---
            for ator in lista_atores[:]:
                if ator.type_id.startswith('vehicle'):
                    try:
                        if ator.is_alive:
                            loc = ator.get_location()
                            dist_ao_cruzamento = loc.distance(posicao_cruzamento)
                            if dist_ao_cruzamento > raio_despawn or loc.z < -10.0:
                                ator.destroy()
                                if ator in lista_atores:
                                    lista_atores.remove(ator)
                    except RuntimeError:
                        if ator in lista_atores:
                            lista_atores.remove(ator)

            world.tick()
            contador_frames += 1
            
            if frame_atual[0] is not None:
                cv2.imshow("Monitoramento de Trafego - Cruzamento Limpo", frame_atual[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nFinalizando via teclado...")
    finally:
        print(f"\nLimpando atores...")
        camera.stop()
        
        for ator in lista_atores:
            try:
                if ator.is_alive:
                    ator.destroy()
            except RuntimeError:
                pass 
                
        cv2.destroyAllWindows()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Mundo restaurado com sucesso.")