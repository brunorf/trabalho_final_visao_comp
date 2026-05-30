import carla
import sys
import time


import numpy as np
import cv2

def processar_imagem_camera(dados_imagem, lista_janela):
    """
    Função de callback disparada a cada frame gerado pelo CARLA.
    Converte os dados brutos de pixels do simulador para o formato do OpenCV (BGRA).
    """
    # Converte o array de bytes brutos em um array unidimensional do NumPy
    array_bruto = np.frombuffer(dados_imagem.raw_data, dtype=np.uint8)
    
    # Redimensiona para o formato de imagem (Altura, Largura, 4 canais: BGRA)
    imagem_bgra = array_bruto.reshape((dados_imagem.height, dados_imagem.width, 4))
    
    # Guarda o frame mais recente em uma estrutura compartilhada para o loop principal ler
    lista_janela[0] = imagem_bgra

def instanciar_camera(world, bp_lib, localizacao_base):
    """
    Spawna a câmera de tráfego calibrada no cruzamento com novos ângulos.
    """
    # 1. Configura o blueprint da câmera RGB
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')  # Mantém o FOV de 40 graus estrito
   

    """
    Location: x=327.28, y=138.37, z=4.06
    Rotation: pitch=0.00, yaw=43.20, roll=0.00
    
    """

    # 2. Define a rotação e translação modificadas pelo usuário
    # Altura mantida em Z=6.0m (altura padrão de postes)
    # Pitch = -15.0 (inclinada 15 graus para baixo)
    # Yaw = 30.0 (rotacionada 30 graus para a direita)
    transform_camera = carla.Transform(
        carla.Location(x=327.28, y=138.37, z=4.0),
        carla.Rotation(pitch=0.0, yaw=45.0, roll=0.0)
    )
    
    # 3. Spawna o ator no mundo do CARLA
    camera_actor = world.spawn_actor(camera_bp, transform_camera)
    print(f"Câmera rotacionada: Pitch={transform_camera.rotation.pitch}, Yaw={transform_camera.rotation.yaw}")
    
    return camera_actor

def configurar_mundo():
    try:
        # 1. Conexão com o Servidor do CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        world = client.get_world()
        if not world.get_map().name.endswith('Town01'):
            print("Carregando Town01...")
            world = client.load_world('Town01')
            
        blueprint_library = world.get_blueprint_library()
        
        # 2. Configuração Estrita do Modo Síncrono
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        # CORREÇÃO AQUI: get_trafficmanager tudo junto, sem o underline (_)
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        
        print("CARLA conectado e configurado com sucesso (TM corrigido).")
        return client, world, blueprint_library, traffic_manager
        
    except Exception as e:
        print(f"Erro na configuração: {e}")
        sys.exit(1)

def obter_semaforos_cruzamento(world, localizacao_central):
    """
    Varre o mapa e encontra os semáforos em um raio de 30 metros do ponto escolhido.
    """
    todos_semaforos = world.get_actors().filter('traffic.traffic_light')
    semaforos_do_cruzamento = []
    
    for semaforo in todos_semaforos:
        # Calcula a distância entre o semáforo atual e o centro do cruzamento
        distancia = semaforo.get_location().distance(localizacao_central)
        
        if distancia < 30.0:  # Raio de 30 metros
            print(semaforo.id)
            semaforo.freeze(True)  # Congela o automatismo do CARLA
            semaforos_do_cruzamento.append(semaforo)
            
    print(f"Sucesso: {len(semaforos_do_cruzamento)} semáforos encontrados e controlados neste cruzamento.")
    return semaforos_do_cruzamento


if __name__ == "__main__":
    client, world, bp_lib, tm = configurar_mundo()
    
    # Coordenadas base do seu cruzamento no Town01
    posicao_cruzamento = carla.Location(x=328.7, y=137.2, z=2.0)
    
    # 1. Toma o controle dos semáforos
    semaforos_locais = obter_semaforos_cruzamento(world, posicao_cruzamento)
    
    # 2. Cria a câmera de monitoramento
    camera = instanciar_camera(world, bp_lib, posicao_cruzamento)
    
    # Lista compartilhada para armazenar os frames da câmera em tempo real
    frame_atual = [None]
    
    # Diz para a câmera enviar cada frame novo para a nossa função de processamento
    camera.listen(lambda dados: processar_imagem_camera(dados, frame_atual))
    
    # Lista de atores criados para garantir a limpeza ao fechar o script
    atores_criados = [camera]
    
    try:
        contador_frames = 0
        print("\nJanela de visualização abrindo. Pressione 'q' na janela ou Ctrl+C no terminal para sair.")
        
        while True:
            # Lógica tradicional de semáforo Round-Robin (muda a cada 10 segundos / 200 frames)
            ciclo = (contador_frames // 200) % 2
            for semaforo in semaforos_locais:
                if ciclo == 0:
                    semaforo.set_state(carla.TrafficLightState.Green)
                else:
                    semaforo.set_state(carla.TrafficLightState.Red)
            
            # Avança o mundo físico em 1 passo (0.05 segundos)
            world.tick()
            contador_frames += 1
            
            # Se a câmera já processou e entregou um frame, exibe na tela via OpenCV
            if frame_atual[0] is not None:
                cv2.imshow("Monitoramento de Trafego - CFTV", frame_atual[0])
            
            # Se o usuário apertar a tecla 'q' dentro da janela do OpenCV, encerra o loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nSimulação interrompida.")
        
    finally:
        # --- LIMPEZA DE SEGURANÇA ---
        print("\nLimpando sensores e restaurando configurações...")
        
        # Desliga a escuta da câmera e destrói o ator do CARLA
        camera.stop()
        for ator in atores_criados:
            if ator.is_alive:
                ator.destroy()
                
        # Fecha as janelas do OpenCV
        cv2.destroyAllWindows()
        
        # Devolve o controle do mundo para o modo assíncrono padrão do CARLA
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Ambiente finalizado com sucesso.")
