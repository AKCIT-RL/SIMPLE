# Pipeline Industrial IH — Resumo

Adaptação da cena `ref_map/IH_basic.usda` (montada no Isaac Sim) para uma task
completa do SIMPLE: teleoperação com física no MuJoCo, gravação em LeRobot para
treino de VLA, e replay com renderização fotorrealista no Isaac Sim.

Este documento é o **resumo do resultado**. O histórico detalhado de decisões,
investigações e correções está em
[`INDUSTRIAL_IH_TASK_PROGRESS.md`](./INDUSTRIAL_IH_TASK_PROGRESS.md).

---

## 1. A task

**`simple/G1IndustrialSortingTeleop-v0`** — sortimento industrial com código de
cores, teleoperado pelo G1 via Meta Quest 3.

> *"put 2 screwdrivers in the blue tote and 1 screw in the red tote, then place
> the blue tote on the right shelf and the red tote on the left shelf."*

- **Peças** sobre a bancada: parafusos e chaves de fenda (graspnet).
- **Totes** vermelho e azul, coloridos nos **dois** engines.
- **Estantes** esquerda e direita (dois carrinhos com colisão de malha real).
- **Prompt e sucesso variam por episódio** (quantidades sorteadas).

### Cena

| Elemento | Papel | Colisão |
|---|---|---|
| TableTrolley_B02_01 | bancada de montagem | malha real (32 cascos) |
| MobileShelvingCart_C05_**left** | prateleira esquerda (destino do tote vermelho) | malha real (48) |
| MobileShelvingCart_C05_**right** | prateleira direita (destino do tote azul) | malha real (48) |
| `table` (primitiva) | superfície funcional do spatial DR | box, topo casado ao tampo do trolley (0.854 m) |
| bin_b04_red / bin_b04_blue | totes | cascos convexos |
| graspnet: parafusos, chaves | peças a sortear | cascos convexos |

A **GravityShelfBinOrganizer** foi removida por performance (213k verts / 392k
faces na malha visual, contra 23k por carrinho).

### Domain randomization
- **Quantidade de peças:** 1–4 por classe, por episódio.
- **Quantidade pedida:** `X ∈ [1, n_visível]` por classe → entra no prompt **e**
  no critério de sucesso.
- **Iluminação:** 3000–8000 K, intensidade 3e4–8e4 (única variação visual —
  materiais ficam nas texturas autorais).

### Recompensa (0.25 por condição; sucesso em 1.0)
`[≥X parafusos no tote vermelho] + [vermelho na prateleira ESQUERDA] +
[≥Y chaves no tote azul] + [azul na prateleira DIREITA]`

Tudo avaliado em estado **vivo** do MuJoCo (posições + contatos geom↔geom).

---

## 2. Arquitetura

```
IH_basic.usda ──► extração offline ──► assets MuJoCo + poses
                                            │
                                            ▼
                            TELEOP (MuJoCo, física real)
                            teleop_decoupled_wbc.py --record
                                            │
                                            ▼
                            dataset LeRobot (+ environment_config)
                                            │
                            ┌───────────────┴───────────────┐
                            ▼                               ▼
                   treino VLA (pi0)              RENDER (Isaac Sim)
                postprocess_psi0_sonic.py     render_decoupled_wbc.py
```

**Princípio central:** o MuJoCo é a fonte da física e o Isaac é a fonte da
imagem. O robô nunca sai do spawn canônico (a stack WBC é calibrada para ele);
a cena é que se adapta.

---

## 3. Ferramentas criadas (`scripts/industrial/`)

| Script | Função |
|---|---|
| `extract_ih_layout.py` | Parseia o `.usda`, converte quaternions→yaw, classifica prims e emite `ih_layout.json` (poses + bloco `simple_frame`) |
| `_usd_common.py` | Helpers USD compartilhados: bootstrap do `pxr` **sem** subir o Kit, tesselação, VHACD, auto-detecção de unidade |
| `extract_furniture_mesh.py` | `USD → OBJ visual → VHACD → MJCF` para mobília estática (+ `_scene.xml` visualizável e `_attach.xml`) |
| `extract_part_mesh.py` | Idem para peças dinâmicas (+ `stable_poses.npy`) |
| `make_shelf_variants.py` | Clona o carrinho em variantes `left`/`right` renomeando só identificadores (malhas compartilhadas) |
| `validate_furniture_mujoco.py` | Integridade de colisão da mobília no MuJoCo |
| `validate_parts_mujoco.py` | Peças via API real do AssetManager + teste de queda |
| `validate_sorting_task.py` | **Validação da task ponta a ponta** (ver §5) |

---

## 4. Como rodar

As malhas convertidas da mobília são **versionadas via Git LFS** (~7 MB), então
não é preciso reconverter nada — basta `git lfs install && git lfs pull`. Guia
completo do usuário: [`docs/G1_INDUSTRIAL_SORTING_GUIDE.md`](./docs/G1_INDUSTRIAL_SORTING_GUIDE.md).

```bash
# 1) Teleoperar e gravar  (Meta Quest 3 na mesma rede; abrir https://<IP>:8012)
.venv/bin/python src/simple/cli/teleop_decoupled_wbc.py simple/G1IndustrialSortingTeleop-v0 \
    --sim-mode=mujoco --record --no-headless

# 2) Renderizar os episódios gravados no Isaac Sim
.venv/bin/python src/simple/cli/render_decoupled_wbc.py simple/G1IndustrialSortingTeleop-v0 \
    --data-dir data/teleop_decoupled_wbc/simple/G1IndustrialSortingTeleop-v0/level-0 \
    --record --save-dir data/render_decoupled_wbc

# 3) Validar a task sem VR/Isaac (rápido)
.venv/bin/python scripts/industrial/validate_sorting_task.py
```

**Sem `--record` o episódio nunca reseta por sucesso:** toda a máquina de estados
(detecção de sucesso + auto-reset) vive dentro do bloco de gravação.

Ajuste de cor dos totes: `Totes_Variants` em `src/simple/assets/totes.py` —
vale para os **dois** engines.

---

## 5. Validação

`validate_sorting_task.py` roda offline (sem VR, sem Isaac) e cobre:

1. import + registro do env;
2. poses de mobília e totes;
   **2c.** *prova* de que o transform relativo robô↔mobília é idêntico ao arranjo
   de operador (tolerância 1e-9);
3. cena MuJoCo real via `gym.make` + `reset` (corpos, cores, pose do robô);
   **3b/3c.** DR de quantidade em `[1, visível]`, refletida no prompt, e
   *roundtrip* de replay (peças + prompt + limiares idênticos);
   **3d.** invariante de render: todo `ObjectActor` tem `.material`;
4. recompensa: objetivo → **1.0**, prateleiras trocadas → **0.5**, uma peça a
   menos → **0.75** (determinístico).

A renderização foi validada **medindo pixels** em frames renderizados de fato
(não por inspeção visual).

---

## 6. Decisões de projeto que valem lembrar

**Robô fixo, cena móvel.** Girar o robô 180° para a posição de operador quebrou o
VR. A solução foi autorar a cena no *frame do operador* e mapeá-la rigidamente
sobre o spawn canônico — rotação de 180° em torno do ponto médio das duas
posições do robô. Isso preserva **exatamente** a geometria relativa
robô↔mobília (provado no estágio 2c).

**Mobília estática pelo caminho articulated.** No MuJoCo ela é anexada como
`ArticulatedObjectActor` de 0 juntas (`MjSpec.from_file` + `attach`); no Isaac,
como prop estático (`add_static_prop`), porque `SingleArticulation` exige uma
articulação real.

**Padding de objetos.** O spawner sempre cria 4+4 instâncias para manter
`observation.object_poses` com shape constante (exigência do LeRobot), parkando
as não usadas abaixo do piso e **sem colisão**.

**Cor dos totes vem do `rgba` do asset**, aplicado por engine: tint de geom no
MuJoCo, material OmniPBR vinculado no Isaac.

---

## 7. Armadilhas encontradas (e que voltariam a morder)

| Sintoma | Causa real |
|---|---|
| Robô girava 180° ao estabilizar | `navigate_cmd[3]` é yaw **absoluto**; `DEFAULT_NAV_CMD` mandava 0 |
| Comportamento inconsistente entre episódios | Objetos de padding em `z=-10` **colidiam** com o chão (plano infinito = semi-espaço) e eram ejetados a ~143 m/s pelo workspace |
| Robô manipulando objetos invisíveis no render | Poses gravadas mapeadas por `asset.name` — que **repete** entre cópias (`metal_screw`×4) — em vez do label da junta; 7 dos 10 objetos recebiam a pose errada, e o padding estacionado sobrescrevia peças visíveis |
| Prompt igual em todo episódio renderizado | `tasks.jsonl[0]` congelado na criação do exporter, em vez do prompt gravado por episódio |
| Segfault ao montar a cena | Arena do MuJoCo estourava → `mjSpec.memory = 256 MB` |
| Episódio não resetava no sucesso | Rodou sem `--record` |
| Prompt errado no dataset | `frame["task"]` é rejeitado pelo `validate_frame`; o certo é `exporter.task` |
| Mobília invisível no render | USDs em **centímetros**; `add_reference_to_stage` não converte unidades |
| Mobília sem textura | Materiais MDL são relativos à origem do asset → referenciar a **URL remota** |
| Totes cinzas | Colorir o material do próprio asset não afeta o render; é preciso **vincular** um OmniPBR novo — e **no fim do reset**, não durante a construção da cena |
| Segfault ao colorir | Vincular material em *instance proxy* é ilegal → des-instanciar antes |

---

## 8. Pendências

- **Executar a teleop e o render da versão final** (v2.3 + cores saturadas) numa
  passada longa, para confirmar estabilidade ao longo de vários episódios.
- **Treino do pi0** com o dataset gerado — o formato foi verificado
  (prompts por episódio são mesclados e remapeados corretamente pelo
  `postprocess_psi0_sonic.py`), mas ainda não houve um treino de fato.
- **Reset por sucesso sem `--record`**: hoje só existe em modo gravação; se for
  útil para depuração, é um pequeno ramo no CLI.
- **`bin_b02`**: é um asset SimReady multi-arquivo e ficou de fora; o `bin_b04`
  cobre os dois totes via variantes de cor.
