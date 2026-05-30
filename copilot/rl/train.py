import os
import signal
import sys
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from semaforo_env import SemaforoInteligenteEnv

# Variável global para lidar com o CTRL+C com segurança
vec_env_global = None
modelo_global = None
models_dir_global = None

def signal_handler(sig, frame):
    print("\n⚠️ CTRL+C detectado. Limpando todos os ambientes do cluster...")
    global vec_env_global, modelo_global, models_dir_global
    # Tenta salvar um checkpoint de emergência para não perder progresso.
    try:
        if modelo_global is not None and models_dir_global is not None:
            modelo_global.save(f"{models_dir_global}/modelo_interrupt")
            if vec_env_global is not None:
                vec_env_global.save(f"{models_dir_global}/vec_normalize_interrupt.pkl")
            print("💾 Checkpoint de interrupção salvo com sucesso.")
    except Exception as e:
        print(f"⚠️ Não foi possível salvar checkpoint de interrupção: {e}")

    if vec_env_global is not None:
        vec_env_global.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def make_env(host_carla, porta_carla, porta_tm, rank):
    def _init():
        env = SemaforoInteligenteEnv(host_carla=host_carla, porta_carla=porta_carla, porta_tm=porta_tm)
        if rank != 0:
            env.verbose = False
        return env
    return _init

def _listar_checkpoints(models_dir):
    """Retorna lista [(idx, caminho_modelo_zip, caminho_vecnorm)] ordenada por idx."""
    pattern = re.compile(r"^modelo_cluster_(\d+)\.zip$")
    checkpoints = []
    for nome in os.listdir(models_dir):
        m = pattern.match(nome)
        if not m:
            continue
        idx = int(m.group(1))
        caminho_modelo = os.path.join(models_dir, nome)
        caminho_vec = os.path.join(models_dir, f"vec_normalize_cluster_{idx}.pkl")
        if os.path.exists(caminho_vec):
            checkpoints.append((idx, caminho_modelo, caminho_vec))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    global vec_env_global, modelo_global, models_dir_global
    models_dir = "./modelos/ppo_semaforo_cluster"
    logdir = "./logs"
    models_dir_global = models_dir
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    # ---------------------------------------------------------
    # MAPEAMENTO DO CLUSTER
    # ---------------------------------------------------------
    ip_n001 = "127.0.0.1" # A própria máquina
    ip_n002 = "10.0.2.2" # Ex: 192.168.1.50
    
    env_fns = []
    
    # Configura 16 ambientes locais (n001-gpu)
    for i in range(16):
        p_carla = 2000 + (i * 5)
        p_tm = 8000 + i
        env_fns.append(make_env(ip_n001, p_carla, p_tm, rank=len(env_fns)))
        
    # Configura 4 ambientes remotos (n002-gpu)
    for i in range(4):
        p_carla = 2000 + (i * 5)
        p_tm = 8100 + i # Usa portas TM diferentes para não dar conflito local
        env_fns.append(make_env(ip_n002, p_carla, p_tm, rank=len(env_fns)))

    print(f"🚀 Iniciando {len(env_fns)} instâncias distribuídas no Cluster...")

    vec_env_global = SubprocVecEnv(env_fns)
    # Necessário para aparecer rollout/ep_rew_mean e rollout/ep_len_mean no TensorBoard.
    vec_env_global = VecMonitor(vec_env_global)

    # Resume automático: se existir checkpoint, carrega modelo e estatísticas do VecNormalize.
    checkpoints = _listar_checkpoints(models_dir)
    if checkpoints:
        idx, caminho_modelo, caminho_vec = checkpoints[-1]
        print(f"♻️ Retomando do checkpoint {idx}: {os.path.basename(caminho_modelo)}")
        vec_env_global = VecNormalize.load(caminho_vec, vec_env_global)
        vec_env_global.training = True
        vec_env_global.norm_reward = False
        modelo = PPO.load(caminho_modelo, env=vec_env_global, device="cuda:0")
        ciclo_inicial = idx + 1
    else:
        vec_env_global = VecNormalize(vec_env_global, norm_obs=True, norm_reward=False, clip_obs=100.0)

        print("🧠 Construindo PPO para o Mega-Cluster...")
        modelo = PPO(
            "MlpPolicy", vec_env_global, verbose=1, tensorboard_log=logdir,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=512, # Aumentei o batch size pois você tem muita VRAM e muitos dados entrando!
            ent_coef=0.01, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            device="cuda:0" # A rede neural treina na GPU 0 da n001 enquanto as outras cuidam da física
        )
        ciclo_inicial = 1

    modelo_global = modelo

    TIMESTEPS = 20000 
    CICLOS = 20

    try:
        for i in range(ciclo_inicial, CICLOS + 1):
            print(f"\n📦 Ciclo {i}/{CICLOS} do Mega-Cluster")
            modelo.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_MultiNode")
            modelo.save(f"{models_dir}/modelo_cluster_{i}")
            vec_env_global.save(f"{models_dir}/vec_normalize_cluster_{i}.pkl")
    finally:
        if vec_env_global is not None:
            vec_env_global.close()

if __name__ == "__main__":
    main()