import carla
import random
import sys

# ==============================================================================
# FUNÇÕES DE CONFIGURAÇÃO E INFRAESTRUTURA
# ==============================================================================

def configurar_mundo():
    """Conecta ao CARLA e avisa o Traffic Manager para respeitar o Mestre"""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        if not world.get_map().name.endswith('Town01'):
            print("Carregando Town01...")
            world = client.load_world('Town01')
            
        bp_lib = world.get_blueprint_library()
        
        tm = client.get_trafficmanager(8000)
        
        # --- A CORREÇÃO ESTÁ AQUI ---
        # O Mundo é controlado pelo train.py, mas o TM precisa saber que está em modo síncrono!
        tm.set_synchronous_mode(True) 
        
        tm.set_global_distance_to_leading_vehicle(1.5) 
        tm.set_random_device_seed(0)
        
        print("[SISTEMA SLAVE] Conectado. Traffic Manager configurado e Síncrono.")
        return client, world, bp_lib, tm
    except Exception as e:
        print(f"Erro na configuração do mundo: {e}")
        sys.exit(1)

# ==============================================================================
# LOOP PRINCIPAL DA SIMULAÇÃO DE TRÁFEGO
# ==============================================================================

if __name__ == "__main__":
    client, world, bp_lib, tm = configurar_mundo()
    
    posicao_cruzamento = carla.Location(x=323.0, y=320.0, z=2.0)
    lista_atores = []
    
    spawn_points = world.get_map().get_spawn_points()
    
    ID_AMBULANCIA = 34      
    ID_OPOSTO = 152         
    ID_PERPENDICULAR = 92   
    
    ponto_spawn_ambulancia = spawn_points[ID_AMBULANCIA]     
    ponto_spawn_oposto = spawn_points[ID_OPOSTO]          
    ponto_spawn_perpendicular = spawn_points[ID_PERPENDICULAR]
    
    distancias_spawns = [
        ponto_spawn_ambulancia.location.distance(posicao_cruzamento),
        ponto_spawn_oposto.location.distance(posicao_cruzamento),
        ponto_spawn_perpendicular.location.distance(posicao_cruzamento)
    ]
    raio_despawn = max(distancias_spawns) + 20.0
    
    bp_carros = [
        bp_lib.find('vehicle.audi.a2'), bp_lib.find('vehicle.audi.etron'),
        bp_lib.find('vehicle.audi.tt'), bp_lib.find('vehicle.bmw.grandtourer'),
        bp_lib.find('vehicle.chevrolet.impala'), bp_lib.find('vehicle.citroen.c3'),
        bp_lib.find('vehicle.ford.crown'), bp_lib.find('vehicle.ford.mustang'),
        bp_lib.find('vehicle.mercedes.coupe'), bp_lib.find('vehicle.micro.microlino'),
        bp_lib.find('vehicle.nissan.micra'), bp_lib.find('vehicle.nissan.patrol'),
        bp_lib.find('vehicle.seat.leon'), bp_lib.find('vehicle.tesla.model3'),
        bp_lib.find('vehicle.toyota.prius'), bp_lib.find('vehicle.ford.ambulance'),
        bp_lib.find('vehicle.volkswagen.t2'), bp_lib.find('vehicle.kawasaki.ninja'),
        bp_lib.find('vehicle.vespa.zx125'), bp_lib.find('vehicle.yamaha.yzf'),
        bp_lib.find('vehicle.bh.crossbike'), bp_lib.find('vehicle.diamondback.century'),
        bp_lib.find('vehicle.gazelle.omafiets')
    ]
    bp_ambulancia = bp_lib.find('vehicle.ford.ambulance')
    
    contador_frames = 0
    frequencia_spawn_carros = 8
    frequencia_ambulancia = 800      
    max_veiculos_simultaneos = 120
    ambulancia_pendente = False      
    
    print(f"\n[INFO] Aguardando o Agente RL (train.py) iniciar o relógio...")
    try:
        while True:
            # ---> O COMANDO MÁGICO <---
            # Trava o script até que o train.py rode o world.tick()
            world.wait_for_tick()
            
            veiculos_ativos = [a for a in lista_atores if a.is_alive and a.type_id.startswith('vehicle')]
            ambulancias_ativas = [a for a in veiculos_ativos if 'ambulance' in a.type_id]
            
            # --- GATILHO DA AMBULÂNCIA ---
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
                    tm.set_route(ambulancia, ["Straight", "Straight"]) 
                    
                    lista_atores.append(ambulancia)
                    ambulancia_pendente = False                   
                    print(f"[EMERGÊNCIA] Ambulância {ambulancia.id} ativa e marchando.\n")

            # --- FASE 3: SPAWN DE VEÍCULOS COMUNS ---
            if contador_frames % frequencia_spawn_carros == 0 and len(veiculos_ativos) < max_veiculos_simultaneos:
                id_sorteado = random.choice([ID_AMBULANCIA, ID_OPOSTO, ID_PERPENDICULAR])
                if ambulancia_pendente and id_sorteado == ID_AMBULANCIA:
                    id_sorteado = random.choice([ID_OPOSTO, ID_PERPENDICULAR])
                    
                ponto_origem = spawn_points[id_sorteado]
                carro_bp = random.choice(bp_carros)
                
                carro = world.try_spawn_actor(carro_bp, ponto_origem)
                if carro is not None:
                    # ---> AS LINHAS RESTAURADAS QUE FIZERAM FALTA <---
                    carro.set_simulate_physics(True)
                    luzes = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
                    carro.set_light_state(carla.VehicleLightState(luzes))
                    carro.set_target_velocity(ponto_origem.get_forward_vector() * 5.0)
                    
                    carro.set_autopilot(True, tm.get_port())
                    tm.ignore_lights_percentage(carro, 0.0)
                    tm.auto_lane_change(carro, False)
                    tm.set_desired_speed(carro, 12.0) 
                    
                    lista_atores.append(carro)

            # --- LIMPEZA DE DESPAWN DINÂMICO BLINDADA ---
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

            contador_frames += 1
                
    except KeyboardInterrupt:
        print("\nFinalizando via teclado...")
    finally:
        print(f"\nLimpando atores...")
        for ator in lista_atores:
            try:
                if ator.is_alive:
                    ator.destroy()
            except RuntimeError:
                pass 
        print("Mundo restaurado com sucesso.")