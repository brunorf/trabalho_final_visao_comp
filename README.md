# Projeto de Semaforo Inteligente com CARLA + RL

Este repositorio contem scripts para simulacao de um cruzamento em T no CARLA, com controle de semaforo por Reinforcement Learning e prioridade para ambulancias.

## Estrutura principal

- `test_integration.py`: script de demonstracao/inferencia com modelo treinado.
- `calibrar_spawns.py`: utilitario para calibrar spawns por coordenada usando spectator e salvar em `spawn_refs.json`.
- `copilot/rl/semaforo_env.py`: ambiente Gymnasium para treino com CARLA.
- `copilot/rl/train.py`: script de treino PPO distribuido.
- `spawn_refs.json`: referencias de spawn calibradas (arquivo local de configuracao).

## Requisitos

- Python 3.10+
- CARLA em execucao
- Dependencias Python (ex.: `stable-baselines3`, `gymnasium`, `numpy`, `carla`)

## Fluxo recomendado

1. Inicie o CARLA.
2. (Opcional) Rode `calibrar_spawns.py` para gerar/atualizar `spawn_refs.json`.
3. Rode `test_integration.py` para validar a logica de controle.
4. Rode `copilot/rl/train.py` para treinar um novo modelo.

## Versionamento

Este projeto possui `.gitignore` preparado para nao versionar:

- pesos de modelos (`*.pt`, `*.pkl`, etc.),
- datasets e saídas pesadas (`runs/`, `logs/`, `modelos/`, etc.),
- imagens locais e arquivos temporarios.

Assim, o repositorio fica leve e focado em codigo/configuracoes.
