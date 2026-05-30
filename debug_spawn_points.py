import carla
import time

client = carla.Client('localhost', 2000)
world = client.get_world()
debug = world.debug

spawn_points = world.get_map().get_spawn_points()

print("Desenhando IDs dos pontos de spawn no simulador por 60 segundos...")
print("Voe com o Spectator perto do seu cruzamento para ler os números no chão!")

for i, ponto in enumerate(spawn_points):
    # Desenha o número do índice do spawn point no ar, na coordenada real
    debug.draw_string(
        ponto.location, 
        str(i), 
        draw_shadow=False,
        color=carla.Color(r=0, g=255, b=0), 
        life_time=60.0
    )
    # Desenha uma seta mostrando a direção que o carro vai nascer andando
    debug.draw_arrow(
        ponto.location, 
        ponto.location + ponto.get_forward_vector() * 2.0, 
        color=carla.Color(r=255, g=0, b=0), 
        life_time=60.0
    )

time.sleep(60.0)
