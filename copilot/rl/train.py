import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from semaforo_env import SemaforoInteligenteEnv

# Variável global para lidar com o CTRL+C com segurança
vec_env_global = None

def signal_handler(sig, frame):
    print("\n⚠️ CTRL+C detectado. Limpando todos os ambientes do cluster...")
    global vec_env_global
    if vec_env_global is not None: vec_env_global.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def make_env(host_carla, porta_carla, porta_tm, rank):
    def _init():
        env = SemaforoInteligenteEnv(host_carla=host_carla, porta_carla=porta_carla, porta_tm=porta_tm)
        if rank != 0: env.verbose = False 
        return env
    return _init

def main():
    global vec_env_global
    models_dir = "./modelos/ppo_semaforo_cluster"
    logdir = "./logs"
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
    
    TIMESTEPS = 20000 
    CICLOS = 20
    
    try:
        for i in range(1, CICLOS + 1):
            print(f"\n📦 Ciclo {i}/{CICLOS} do Mega-Cluster")
            modelo.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_MultiNode")
            modelo.save(f"{models_dir}/modelo_cluster_{i}")
            vec_env_global.save(f"{models_dir}/vec_normalize_cluster_{i}.pkl")
    finally:
        if vec_env_global is not None: vec_env_global.close()

if __name__ == "__main__":
    main()