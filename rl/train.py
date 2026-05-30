import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

# Importação local corrigida (estando na mesma pasta 'rl')
from semaforo_env import SemaforoInteligenteEnv

def main():
    print("Inicializando o ambiente...")
    env = SemaforoInteligenteEnv()
    check_env(env, warn=True) 

    # SUBINDO UM NÍVEL (../) para salvar na raiz do projeto, fora da pasta rl
    models_dir = "./modelos/ppo_semaforo"
    logdir = "./logs"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    print("Construindo a Rede Neural PPO...")
    modelo = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        learning_rate=0.0003,
        n_steps=2048,
        ent_coef=0.05,
        device="cpu"
    )

    TIMESTEPS_POR_CICLO = 10000 
    CICLOS_TOTAIS = 10 

    print("Iniciando a Aprendizagem por Reforço...")
    for i in range(1, CICLOS_TOTAIS + 1):
        modelo.learn(total_timesteps=TIMESTEPS_POR_CICLO, reset_num_timesteps=False, tb_log_name="PPO_Semaforo")
        caminho_modelo = f"{models_dir}/modelo_ciclo_{i}"
        modelo.save(caminho_modelo)
        print(f"Checkpoint salvo: {caminho_modelo}")

if __name__ == "__main__":
    main()