# Guia — Teleoperar e renderizar a task `G1WholebodyLocomotionPickTotesShelfToTableMirror`

Como coletar dados e renderizar a task de **entrega direcionada por linguagem**: o G1
pega a tote azul da estante e a leva para a **mesa da esquerda** ou **da direita**,
conforme o prompt do episódio — usando a **mão** que o prompt mandar.

**A tarefa:** duas mesas idênticas ficam nos extremos opostos do corredor (uma passando
o extremo −X, outra o extremo +X). Quando o robô está de frente para as estantes no
spawn, uma mesa fica à sua **esquerda** e a outra à **direita**. Como ele não vê nenhuma
das mesas até girar para um dos lados, a decisão de para onde ir vem **só do prompt** —
é isso que a task mede (generalização por linguagem).

> *"Pick up the blue tote from the shelf with your left hand and place it on the right table."*

Variante de [`g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop`](../src/simple/tasks/g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop.py);
o código da task está em
[`src/simple/tasks/g1_wholebody_locomotion_pick_totes_shelf_to_table_mirror_teleop.py`](../src/simple/tasks/g1_wholebody_locomotion_pick_totes_shelf_to_table_mirror_teleop.py).

---

## 1. Pré-requisitos

Iguais aos das outras tasks de teleop — **não repita aqui, siga os guias base**:

- **Ambiente** (`.venv`, submódulos, `git lfs pull`): [`source/tutorials/installation.md`](source/tutorials/installation.md).
- **Certificados SSL do televuer + conexão do headset** (WebXR/HTTPS, `?ws=wss://...`):
  [`source/tutorials/teleop.md`](source/tutorials/teleop.md), seção **SSL Certificates (televuer)**.
- **Primeiros passos no VR** (botões, ativar braços, andar):
  [`teleop_simple_study/vr_teleop_beginners_guide.md`](teleop_simple_study/vr_teleop_beginners_guide.md).

Na primeira execução, os assets (corridor0, totes) são baixados/materializados
automaticamente; o pipeline ONNX do WBC também é baixado no primeiro `Enter VR`.

---

## 2. As duas flags do episódio

O prompt de cada episódio é montado a partir de **dois eixos**:

| Flag CLI | Valores | O que controla |
|---|---|---|
| `--pick-hand` | `left` · `right` · `both` · `random` | qual mão pega a tote |
| `--target-side` | `left` · `right` · `random` | em qual mesa entregar |

- **`random` (default)**: sorteia o valor a cada episódio.
- **Fixo** (`left`/`right`/`both`): pina o valor para **coleta controlada** (ver §5).

> **A mão só orienta o comportamento — ela NÃO pontua.** O sucesso é **estrito no lado
> da mesa**: a tote precisa ficar apoiada, em pé, na mesa **do lado comandado**. Entregar
> na mesa errada é falha (é justamente o sinal de generalização). Entregar com a mão
> "errada" não invalida — mas as demos devem seguir a mão do prompt para não misturar
> trajetórias corretas com mãos diferentes.

O prompt (6 combinações) segue o template:

```
Pick up the blue tote from the shelf with {your left hand | your right hand | both hands}
and place it on the {left | right} table.
```

---

## 3. Teleoperar

```bash
cd <repo>              # o worktree/checkout com a task
source .venv/bin/activate   # (no worktree de merge: .venv-merge)

# lado e mão sorteados a cada episódio
python src/simple/cli/teleop_decoupled_wbc.py \
    simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0 \
    --sim-mode=mujoco --record --no-headless

# fixando (ex.: só mesa da direita, mão direita — ver §5 sobre balanceamento)
python src/simple/cli/teleop_decoupled_wbc.py \
    simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0 \
    --sim-mode=mujoco --record --no-headless --target-side right --pick-hand right
```

> **`--record` é obrigatório para o episódio resetar/salvar ao concluir** — a detecção de
> sucesso vive no fluxo de gravação. Também é o que produz os dados para o render.

**No visor (HUD)** aparecem o comando do episódio e o status da entrega:

```
HAND: LEFT
TABLE: RIGHT
delivered: ...        →  "OK" (mesa certa) | "WRONG TABLE" (mesa errada) | "..."
```

**O que fazer**
1. Ative os braços e estabilize (ver guia base). O robô nasce de frente para as estantes.
2. **Pegue a tote AZUL** (é a única marcada; as demais são distratoras) usando a **mão do
   prompt** (`HAND`).
3. **Gire para o lado comandado** (`TABLE`) para encontrar a mesa daquele extremo do
   corredor e caminhe até ela. A mesa do outro lado fica fora do campo de visão — por isso
   a direção tem que sair do prompt.
4. **Apoie a tote em pé** na mesa. Quando `delivered: OK`, o episódio conclui e salva.

**Dados** vão para
`data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0/level-0/sessions/<timestamp>__<operator>/`
(o terminal imprime o caminho exato). Cada episódio grava `pick_hand` e `target_side` no
`metadata`/`state_dict`.

---

## 4. Renderizar

Não precisa de nada especial — o render é genérico e as duas mesas saem automaticamente
(engine Isaac renderiza qualquer primitivo `Box`). Aponte `--data-dir` para a **pasta da
sessão** que o teleop imprimiu:

```bash
# teste rápido: 1 episódio, sem gravar
python src/simple/cli/render_decoupled_wbc.py \
    simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0 \
    --data-dir data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0/level-0/sessions/<ts>__<operator> \
    --num-episodes 1 --no-record

# passada completa, gravando
python src/simple/cli/render_decoupled_wbc.py \
    simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0 \
    --data-dir data/teleop_decoupled_wbc/simple/G1WholebodyLocomotionPickTotesShelfToTableMirrorTeleop-v0/level-0/sessions/<ts>__<operator> \
    --record --save-dir data/render_decoupled_wbc
```

(`--sim-mode` já é `mujoco_isaac` por padrão.) O replay reconstrói o mesmo `pick_hand`/
`target_side` gravados, então o prompt e a mesa-alvo do render batem com a captura.

---

## 5. Coleta de dados e balanceamento

O objetivo é um dataset com **distribuição uniforme** entre os lados (e mãos). Fluxo:

1. **Dados antigos** (da task de mesa única) entram como exemplos de **"left"** — só é
   preciso **relabelar o prompt** para "...left table." no postprocess. Como o robô só vê a
   mesa no fim (ao girar), e as mesas ficam em extremos opostos (nunca as duas no frame),
   a mistura de cenas 1-mesa/2-mesas **não cria atalho visual** para o modelo.
2. **Catch-up**: grave só `--target-side right` até `n_right == n_left`.
3. **Contagem**: conte episódios por `target_side` (gravado no `metadata`) para saber
   quanto falta.
4. **Crescimento uniforme**: depois de empatar, alterne/preencha os lados (conta-e-preenche)
   para manter 50/50.

> **`both` é uma habilidade distinta** (agarre bimanual), não só um rótulo — provavelmente
> precisa de mais exemplos para o policy aprender o agarre com as duas mãos.

No **eval**, o prompt é o condicionante: o sucesso mede entrega no lado comandado. A task
expõe `delivered_to_wrong_table()` como diagnóstico (segue-a-linguagem vs chuta um lado).

---

## 6. Ajustes rápidos

| O quê | Onde |
|---|---|
| Posição da mesa da direita (ponto de espelho em X) | `_CORRIDOR_CENTER_X` na task (default = spawn X do robô) |
| Posição/tamanho da mesa da esquerda | `_TABLE_POSITION_XY` / `_TABLE_SIZE` na **task-pai** (compartilhado) |
| Texto do prompt | `LanguageDRCfg` / `_HAND_CLAUSE` na task (mantenha os slots `{hand_clause}`/`{side}`) |
| Tolerância de "tote em pé" p/ sucesso | `_STABLE_TILT_TOLERANCE_DEG` na task-pai |

---

## 7. Problemas comuns

| Sintoma | O que fazer |
|---|---|
| Episódio não reseta ao concluir | Faltou `--record` |
| As duas mesas aparecem coladas/no mesmo extremo | Espelho errado — deve ser em **X** (`_CORRIDOR_CENTER_X`), não em Y |
| Mesa da direita atravessa/colide com as estantes | Ajuste `_CORRIDOR_CENTER_X` para afastar mais do extremo +X |
| Headset não conecta / erro de certificado | Ver setup base ([teleop.md](source/tutorials/teleop.md)) |
| `delivered: WRONG TABLE` no HUD | Você entregou na mesa do lado oposto ao comandado |
