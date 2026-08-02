# Renderização Isaac Sim com backdrop Warehouse — tote shelf-to-table

Guia de acompanhamento para validar, numa máquina com GPU capaz de rodar
Isaac Sim (RTX 4090 ou RTX 5080), a renderização fotorrealista dos episódios
de teleoperação já gravados da task
`g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop`
(`simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0`), usando o
ambiente "Warehouse" (SimReady) do Isaac Sim como plano de fundo.

Este documento cobre: o que mudou no código, o que instalar na máquina de
destino, como transferir os dados gravados, os comandos exatos para rodar a
renderização, onde ficam os outputs, e como depurar/ajustar o backdrop.

## Contexto e o que foi implementado

A task não usa o mecanismo `layout.scene`/`SceneManager` (HSSD) do
framework — o corredor/prateleiras (`corridor0`) e a mesa são montados
manualmente em `reset()` (ver `src/simple/tasks/g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop.py`,
comentário nas linhas ~123-127). Isso é intencional: o único `SceneManager`
(`hssd`) carrega casas e não produz colisão MuJoCo real para este corredor
industrial. Mais contexto sobre essa decisão está em
`docs/teleop_simple_study/toteweg_factory_scene_migration_plan.md`
(Phase 2) — **essa pasta é local/gitignorada** (`.gitignore:94`), então esse
arquivo pode não existir na máquina de destino após o clone; se precisar
dele, copie manualmente junto com os dados no passo 2.

Como consequência, o `IsaacSimSimulator` (`src/simple/engines/isaacsim.py`)
tinha dois bugs que impediam essa task de renderizar no Isaac Sim:

1. `__update_scene` sempre lia `self.task.layout.scene.uid` — como essa task
   nunca popula `layout.scene`, isso resultava em `AttributeError` logo no
   primeiro reset.
2. O corredor (`corridor0`) é um `StaticObjectActor`
   (`src/simple/core/actor.py`), mas o loop que cria objetos no Isaac só
   aceitava `ObjectActor`/`ArticulatedObjectActor` — `corridor0` era sempre
   ignorado, então mesmo sem o crash acima ele nunca apareceria no Isaac.

Ambos os USD já existiam prontos (`src/simple/assets/fixtures/corridor0/corridor0.usd`,
já referenciado por `AssetManager.get("fixtures").load("corridor0")`) — não
era um problema de asset faltando, só de integração no engine.

Mudanças feitas em `src/simple/engines/isaacsim.py`:

- `__update_scene` agora usa `getattr(self.task.layout, "scene", None)` e,
  se não houver `scene`, aplica só o(s) `table`/`table2` (se existirem) via
  o novo método `__update_tables` e retorna — sem crashar e sem exigir que a
  task adote HSSD/`SceneManager`.
- O loop de objetos em `update_layout` agora inclui `StaticObjectActor`, então
  `corridor0` passa a ser criado/atualizado no Isaac pelo mesmo caminho
  genérico usado para totes/robô (`__create_object`/`__update_object`).
- Novo `_setup_background()` (chamado uma vez em `_setup_scene`): referencia
  um USD externo (o Warehouse) num prim próprio (`/World/background`), como
  backdrop puramente visual — **sem colisão**, já que a física continua 100%
  no MuJoCo (`sim_mode=mujoco_isaac`). O caminho do USD é resolvido, em
  ordem: `task.metadata["isaac_background_usd"]` → variável de ambiente
  `SIMPLE_ISAAC_BACKGROUND_USD` → nenhum backdrop (comportamento atual
  preservado para todas as outras tasks).
- Constantes `IsaacSimSimulator.BACKGROUND_PRIM_PATH` /
  `BACKGROUND_POSITION` / `BACKGROUND_ORIENTATION_WXYZ` / `BACKGROUND_SCALE`
  controlam onde o Warehouse é posicionado — hoje na origem, identidade,
  escala 1:1; ajustar por inspeção visual (ver seção "Ajustando o backdrop").

Em `src/simple/cli/render_decoupled_wbc.py` e `replay_decoupled_wbc.py`, nova
opção `--isaac-background-usd <path>` seta essa env var antes de criar o
ambiente — não precisa exportar a variável manualmente se usar a flag.

**Importante**: nada disto foi testado com Isaac Sim de verdade — esta
máquina não tem GPU capaz de rodar Isaac. A validação real é o objetivo deste
guia, na máquina com RTX 4090/5080.

## 1. Preparar a máquina de destino

### 1.1. Clonar o repositório

```bash
git clone <url-do-repo> SIMPLE
cd SIMPLE
git checkout feat/weg_wmo   # ou a branch/PR que contém estas mudanças
git submodule update --init --recursive
```

### 1.2. Instalar o ambiente Python (uv)

Requer Python 3.10 e GPU NVIDIA classe RTX com drivers/CUDA compatíveis com
Isaac Sim 4.5.

```bash
bash scripts/setup_python_env.sh
bash scripts/install_curobo.sh
source .venv/bin/activate
python -c "import simple; print(simple.__version__)"
```

Se estiver usando `robo-nix` em vez de uv, entre com `robo shell` antes de
rodar os dois scripts acima (ver `README.md`/`CLAUDE.md` da raiz do repo).

### 1.3. Confirmar que o Isaac Sim 4.5 está instalado e acessível

`isaacsim[all,extscache]==4.5.0` é dependência base do projeto
(`pyproject.toml:9`), instalada pelo `setup_python_env.sh`. Primeiro
confirme que o pacote está no venv:

```bash
uv pip list --python .venv/bin/python | grep -i isaacsim
```

**Não** teste com `python -c "import omni.isaac.core"` direto — a maior
parte de `omni.*`/`isaacsim.*` só fica importável **depois** que o processo
Kit é iniciado via `SimulationApp(...)` (é ele quem carrega as extensões
dinamicamente; ver `src/simple/engines/isaac_app.py` e
`src/simple/envs/base_dual_env.py:83-88`). Um `import` direto falha com
`ModuleNotFoundError: No module named 'omni.isaac'` mesmo com tudo instalado
corretamente — isso não é sinal de instalação quebrada. O teste real é:

```bash
python -c "
from omni.isaac.kit import SimulationApp
app = SimulationApp({'headless': True})
import omni.isaac.core
import isaacsim.core.prims
print('ok')
app.close()
"
```

(demora dezenas de segundos — o Kit precisa subir o processo inteiro). Se
isso falhar, revise a instalação do Isaac Sim 4.5 antes de prosseguir —
esse passo é independente das mudanças deste guia.

### 1.4. Localizar o USD do ambiente Warehouse (SimReady)

O Isaac Sim 4.5 inclui o ambiente de exemplo "Simple Warehouse" nos assets
padrão da NVIDIA. O caminho exato depende de como os assets estão
configurados na máquina (Nucleus local/remoto vs. cache local baixado pelo
Asset Browser) — **confirme o caminho certo abrindo o Isaac Sim e navegando
até ele pelo Asset Browser** (`Window > Browsers > Assets`, procurar por
`Isaac/Environments/Simple_Warehouse/warehouse.usd`). Formatos típicos:

- Nucleus (servidor local ou o Nucleus público da NVIDIA):
  `omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd`
- Cache local (se os assets foram baixados/mirrorados):
  `/home/<user>/isaacsim_assets/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd`
  (ou caminho equivalente dentro da instalação do Isaac Sim).

Guarde esse caminho — ele vai para `--isaac-background-usd` no passo 3.

## 2. Transferir os dados de teleoperação gravados

`data/` está no `.gitignore` (ver `.gitignore:51`) — os episódios gravados
**não** vêm pelo `git clone`, precisam ser copiados manualmente desta máquina
para a de destino:

```bash
# nesta máquina (origem)
rsync -avz --progress \
  data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/ \
  <user>@<maquina-destino>:SIMPLE/data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/
```

(ou `scp -r`, pendrive, etc. — o que for mais prático). O diretório contém
`level-0/{data,meta,videos}`, formato LeRobot, ~51MB no momento deste guia.

## 3. Rodar a renderização

Sempre valide visualmente com **poucos episódios e sem `--record`** antes de
rodar o dataset completo — cada passo abaixo constrói confiança sobre o
anterior.

### 3.1. Sanity check — sem Warehouse, 1 episódio, sem gravar

Confirma que o crash do `layout.scene` foi corrigido e que `corridor0`
aparece no Isaac (passos 1 e 2 do que foi implementado):

```bash
render-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --data-dir data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0 \
  --sim-mode mujoco_isaac \
  --num-episodes 1 \
  --headless=False
```

Com `--headless=False` (e `--webrtc` ligado por padrão) a viewport do Isaac
deve abrir. Verifique visualmente: sem crash no reset, prateleiras/corredor
(`corridor0`), mesa e totes visíveis, robô se movendo conforme os dados
gravados.

### 3.2. Com o backdrop Warehouse — 1 episódio, sem gravar

```bash
render-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --data-dir data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0 \
  --sim-mode mujoco_isaac \
  --num-episodes 1 \
  --headless=False \
  --isaac-background-usd omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd
```

(troque o caminho pelo confirmado no passo 1.4). Verifique visualmente que o
Warehouse aparece ao redor da cena sem cobrir/deslocar `corridor0`, mesa ou
totes. Se a posição/escala estiver errada (chão do warehouse não alinhado
com o chão da cena, por exemplo), ver "Ajustando o backdrop" abaixo.

### 3.3. Rodar o dataset completo, gravando

Só depois dos dois checks acima passarem:

```bash
render-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --data-dir data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0 \
  --sim-mode mujoco_isaac \
  --headless=True \
  --record \
  --save-dir data/render_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --isaac-background-usd omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd
```

- Omita `--num-episodes` (ou use `-1`, o padrão) para processar todos os
  episódios do `--data-dir`.
- `--headless=True` é recomendado para a passada final (mais rápido, sem
  overhead de viewport).

### Onde ficam os outputs

- **Comando executado a partir de**: raiz do repo (`SIMPLE/`) — todos os
  caminhos acima (`--data-dir`, `--save-dir`) são relativos a ela.
- **Dataset final (LeRobot, com imagens Isaac Sim)**:
  `<--save-dir>/<env_id>/level-0/` (ex.:
  `data/render_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0/level-0/`),
  com a mesma estrutura `{data,meta,videos}` do dataset de entrada — as
  imagens/vídeos de `observation.images.ego_view` são as recém-renderizadas
  pelo Isaac Sim, o resto (ações, proprioceptivo) é copiado do dataset
  original.

## 4. Ajustando o backdrop (se a posição/escala do Warehouse estiver errada)

`IsaacSimSimulator` (`src/simple/engines/isaacsim.py`) expõe as constantes
de classe:

```python
BACKGROUND_PRIM_PATH = "/World/background"
BACKGROUND_POSITION = (0.0, 0.0, 0.0)
BACKGROUND_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)
BACKGROUND_SCALE = (1.0, 1.0, 1.0)
```

Editar esses valores, rodar de novo o passo 3.2 (1 episódio, sem gravar,
viewport visível) e comparar visualmente até o chão do Warehouse coincidir
com o chão de `corridor0`/mesa. Não há necessidade de precisão física — é só
um pano de fundo, sem colisão.

## 5. Checklist de validação

- [ ] O smoke test com `SimulationApp` do passo 1.3 funciona no ambiente da
      máquina de destino.
- [ ] Passo 3.1 roda sem `AttributeError`/crash no reset.
- [ ] `corridor0` (prateleiras/corredor) visível na viewport no passo 3.1.
- [ ] Passo 3.2: Warehouse visível ao redor da cena, sem sobrepor objetos da
      task.
- [ ] Passo 3.3 completa todos os episódios e grava o dataset em
      `--save-dir`.
- [ ] Inspecionar 2-3 vídeos/frames do dataset gravado
      (`<save-dir>/.../videos/chunk-000/observation.images.ego_view/`) para
      confirmar qualidade visual antes de considerar o pipeline pronto.

## 6. Troubleshooting

- **`AttributeError: 'Layout' object has no attribute 'scene'`**: as mudanças
  deste guia não foram aplicadas/mescladas na branch usada na máquina de
  destino — confirme `git log -- src/simple/engines/isaacsim.py` e que o
  checkout inclui o commit com `__update_scene`/`getattr`.
- **`corridor0` não aparece mesmo sem erro**: confirme que
  `StaticObjectActor` está no import de `src/simple/engines/isaacsim.py`
  (`from simple.core.actor import ObjectActor, StaticObjectActor, ...`) e no
  filtro de `update_layout`.
- **Warehouse não aparece / `add_reference_to_stage` falha**: o caminho
  passado em `--isaac-background-usd` está errado ou o Nucleus/servidor de
  assets não está acessível dessa máquina — confirme abrindo o mesmo caminho
  pelo Asset Browser dentro do próprio Isaac Sim primeiro.
- **Erro ao rodar sem GPU / sem Isaac Sim instalado**: esperado — este
  pipeline só roda numa máquina com Isaac Sim 4.5 + GPU RTX de verdade.
- **`ModuleNotFoundError: No module named 'omni.isaac'` num `import` direto**:
  não é erro — `omni.*`/`isaacsim.*` só carregam depois de
  `SimulationApp(...)` iniciar o processo Kit. Não teste com `import
  omni.isaac.core` isolado; use o smoke test do passo 1.3.
- **`curobo` falha ao compilar com `RuntimeError: The detected CUDA version
  (X) mismatches the version that was used to compile PyTorch (12.8)`**: o
  `nvcc` do `PATH` não é a mesma versão usada pelo wheel do PyTorch (índice
  `cu128`, `pyproject.toml:62`). Exporte `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH`
  para uma instalação do CUDA Toolkit 12.8 antes de rodar
  `install_curobo.sh` (ver `docs/source/troubleshooting.md`).
