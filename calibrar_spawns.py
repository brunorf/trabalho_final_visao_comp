import carla
import json
import os


ID_AMBULANCIA = 34
ID_OPOSTO = 152
ID_PERPENDICULAR = 92
ARQUIVO_SAIDA = "spawn_refs.json"


def location_to_dict(loc):
    return {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)}


def calibrar(client_host="127.0.0.1", client_port=2000, timeout=20.0):
    client = carla.Client(client_host, client_port)
    client.set_timeout(timeout)
    world = client.get_world()
    spectator = world.get_spectator()

    print(f"[INFO] Mapa atual: {world.get_map().name}")
    print("[INFO] Este script NAO ativa modo sincrono.")
    print("[INFO] Navegue com o spectator no CARLA e pressione ENTER para capturar cada via.\n")

    ordem = [
        (ID_AMBULANCIA, "Via 34 / ambulancia"),
        (ID_OPOSTO, "Via 152 / oposta"),
        (ID_PERPENDICULAR, "Via 92 / perpendicular"),
    ]

    refs = {}
    for spawn_id, nome in ordem:
        input(f"[CALIB] Posicione o spectator em {nome} e pressione ENTER...")
        tr = spectator.get_transform()
        refs[spawn_id] = carla.Location(
            x=tr.location.x,
            y=tr.location.y,
            z=max(0.5, tr.location.z),
        )
        print(
            f"[CALIB] Capturado {nome}: "
            f"x={refs[spawn_id].x:.2f}, y={refs[spawn_id].y:.2f}, z={refs[spawn_id].z:.2f}"
        )

    data = {
        "map_name": world.get_map().name,
        "references": {
            str(ID_AMBULANCIA): location_to_dict(refs[ID_AMBULANCIA]),
            str(ID_OPOSTO): location_to_dict(refs[ID_OPOSTO]),
            str(ID_PERPENDICULAR): location_to_dict(refs[ID_PERPENDICULAR]),
        },
    }

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Calibracao salva em: {os.path.abspath(ARQUIVO_SAIDA)}")
    print("[OK] Agora rode o test_integration.py e ele vai carregar esse arquivo automaticamente.")


if __name__ == "__main__":
    calibrar()
