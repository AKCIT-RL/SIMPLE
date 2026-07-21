# Nova Task Industrial (IH_basic) — Progresso da Execução (Plano A)

Registro do que foi construído para adaptar o ambiente `ref_map/IH_basic.usda`
(montado no Isaac Sim) para uma nova task industrial no framework SIMPLE.

**Decisões que guiaram a execução** (acordadas antes de codar):

1. **Cenário de render no Isaac Sim:** reusar o `warehouse.usd` genérico já
   integrado (`WarehouseSceneManager`). O `IH_basic.usda` serve como **fonte de
   layout**, não como fundo renderizado.
2. **Objetivo da task:** **sortimento multi-peça** (pegar várias peças e separar
   em totes/bins) — delta de lógica frente às 3 tasks industriais existentes, que
   são de peça única.
3. **Assets sem equivalente MuJoCo** (engrenagem, conector-T, bin_b02): **criar um
   `AssetManager` novo** com malha própria.
4. **Mobília (mesa/prateleira/carrinho):** **Plano A — colisão de malha real**
   (o robô interage de fato com as superfícies). Plano B (colisão por caixas
   primitivas) fica como contingência e **só é acionado com autorização explícita**.

---

## 1. O que foi feito e validado

Foco desta rodada: **a fundação do pipeline de assets (Plano A)** — a parte de
maior risco. Tudo abaixo está implementado e testado ponta a ponta.

### 1.1 Toolchain USD estabelecido (sem bootar o Kit)
O `pxr` (USD) é lido **standalone** a partir das libs que o Isaac Sim já traz
(`isaacsim/extscache/omni.usd.libs-*`), sem iniciar o `SimulationApp`/Kit (que é o
padrão pesado usado em `scripts/export_obj_usd.py`). Basta pôr `omni.usd.libs/bin`
e `.../pxr` no `LD_LIBRARY_PATH` — o extrator faz esse bootstrap sozinho e
re-executa uma vez se necessário. Confirmado: `USD 0.22.11` importando e lendo
USDC binário.

### 1.2 Scripts entregues (`scripts/industrial/`)
| Script | Passo | O que faz |
|---|---|---|
| `extract_ih_layout.py` | 1 | Parseia `IH_basic.usda` (ASCII, sem dependência), converte quaternions→yaw-Z, classifica os 22 prims em furniture/part/container/light, e emite `ih_layout.json` com poses (metros/radianos), escala e URL de cada asset. |
| `extract_furniture_mesh.py` | 2 | `USD → OBJ visual (metros) → VHACD → N cascos convexos → snippet MJCF`. Baixa o USD do S3 (ou usa cache/local), tessela todas as malhas com o transform mundo, escala cm→m (×0.01), decompõe em convexos e gera a mobília como corpo estático (visual `contype=0` + colisão convexa). |
| `validate_furniture_mujoco.py` | 3 | Carrega o MJCF no MuJoCo, solta um tote sobre a peça e checa integridade de colisão: compila, sem tunelamento, contatos limitados, velocidade finita, altura de repouso. Emite veredito PASS / candidato-a-Plano-B por peça. |

### 1.3 Layout extraído (`data/assets/industrial/ih_layout.json`)
22 prims: **3 mobílias**, **4 containers** (bin_b02, 3× bin_b04), **14 peças**
(6 parafusos, 3 engrenagens, 4 conectores-T, 1 longbox), 1 luz.

Poses das mobílias (frame do mundo, o `unitsResolve=0.01` confirma assets em cm):

| Mobília | pos (x,y,z) m | yaw |
|---|---|---|
| TableTrolley_B02_01 (bancada montagem) | (+0.424, −3.293, 0) | 0° |
| GravityShelfBinOrganizer_A02_01 (prateleira) | (+0.821, −1.456, 0) | −127° |
| MobileShelvingCart_C05_01 (carrinho despacho) | (+1.276, −5.342, 0) | +54° |

### 1.3.1 Helpers USD compartilhados (`scripts/industrial/_usd_common.py`)
`_bootstrap_pxr()` (pxr standalone, re-exec do script de entrada uma vez),
`download`, `stage_meters_per_unit` (auto-detecção de unidade), `usd_to_trimesh`
(tessela + bake do transform mundo + escala), `decompose` (VHACD). Os extratores
de mobília e de peça importam daqui — sem duplicação.

### 1.4 Mobílias extraídas + validadas no MuJoCo (Plano A)
Geradas em `data/assets/industrial/<nome>/` (`visuals/*.obj`, `collision/*.obj`,
`*.mjcf.xml`, `*.usd`). `data/` é gitignored → assets são **regeneráveis** pelos
scripts, não versionados (mesmo padrão de graspnet/objaverse).

| Mobília | Tamanho (m) | Cascos convexos | Resultado MuJoCo |
|---|---|---|---|
| TableTrolley | 0.739 × 2.244 × 0.854 | 32 | **PASS** — tote pousa no tampo (z≈0.91), estável, max ncon=1 |
| GravityShelf | 0.973 × 0.689 × 1.685 | 48 | **PASS** — tote pousa no topo (z≈1.74), estável, max ncon=4 |
| MobileShelvingCart | 0.518 × 1.097 × 1.873 | 48 | **PASS** — 5 níveis de prateleira sólidos (z≈0.19/0.60/0.98/1.30/1.66) |

**Nenhuma peça precisou do Plano B.** O risco que eu havia sinalizado — as rampas
anguladas da prateleira gravitacional decomporem mal e desestabilizarem o tote —
**não se materializou**: a colisão da prateleira é limpa e estável. O carrinho, que
é o destino de despacho, tem 5 superfícies horizontais utilizáveis para pousar
totes (mapeadas por varredura de drops).

### 1.5 Peças pegáveis dinâmicas + `IndustrialPartsManager`
Diferente da mobília (estática, colisão soldada ao mundo), as peças são **corpos
rígidos livres** que o motor MuJoCo monta sozinho a partir de
`collision_meshes_mujoco` ([engines/mujoco.py:288](src/simple/engines/mujoco.py#L288):
cascos convexos → geoms + `freejoint`, massa total ~0.1 kg). Então o entregável é o
layout de pasta + o AssetManager, espelhando o `graspnet`/`totes`.

- **`scripts/industrial/extract_part_mesh.py`** — `USD → visual.obj + VHACD
  (convex_piece_*.obj) + stable_poses.npy + <name>.usd`. Diferença crítica vs
  mobília: os props Isaac (Factory/Props) são autorados **em metros**
  (`metersPerUnit=1.0`, ≠ mobília em cm), então a escala é **lida do USD**, nunca
  hardcodada. `stable_poses` seguem a convenção do pipeline
  ([spatial.py:256](src/simple/dr/spatial.py#L256) faz `p[2] = stable_pose[2] +
  surface_height`): só a *orientação* estável vem do `trimesh.compute_stable_poses`;
  o z é recalculado como `-min_z` (apoia a base na superfície, igual ao totes).
- **`src/simple/assets/industrial_parts.py`** — `IndustrialPartsAssetManager`
  registrado como `"industrial_parts"`; nomes `{0: factory_gear_large,
  1: t_connector_physics}`; `uid==label==name` (evita bug de sincronia de nome no
  replay). Exportado em `assets/__init__.py`.

Peças extraídas (`data/assets/industrial_parts/<name>/`, gitignored):

| Peça | metersPerUnit | Tamanho (m) | Cascos | Stable poses | Validação MuJoCo |
|---|---|---|---|---|---|
| factory_gear_large | 1.0 | 0.062×0.062×0.025 | 16 | 4 | **PASS** — reassenta no stable_z exato, sem tunelar |
| t_connector_physics | 1.0 | 0.038×0.114×0.040 | 16 | 4 | **PASS** — idem, estável em 4 orientações |

Validação via `scripts/industrial/validate_parts_mujoco.py`: carrega pela **API
real do `AssetManager`**, monta o corpo como o motor faz, solta de `stable_z+0.15`
e confirma que reassenta no `stable_z` exato (prova que a stable pose é um repouso
genuíno), sem tunelamento/explosão, velocidade final ~0.

```bash
.venv/bin/python scripts/industrial/validate_parts_mujoco.py
```

### 1.6 Integração da mobília na cena MuJoCo — caminho decidido + validado
Pesquisa: `_setup_scene` ([engines/mujoco.py:153](src/simple/engines/mujoco.py#L153))
itera `layout.actors` e despacha por tipo — `_build_object` (dinâmico, com
`add_freejoint`), `_build_articulated_object` (`MjSpec.from_file` + `attach` num
frame na pose), `_build_primitive` (table box). O layout é populado no DR manager.

**Decisão (com o usuário):** mobília estática entra pelo **caminho articulated**
(`ArticulatedObjectActor` com 0 juntas) — reusa `MjSpec.from_file` + `attach` sem
mudar o engine. Cada mobília expõe um `mjcf_path` = `<name>_attach.xml` (full-model
`<mujoco>`, corpo estático soldado, gerado pelo extrator de mobília).

**Validado** (`MjSpec.from_file` + `attach` das 3 mobílias em poses/yaws
arbitrários → 133 geoms compilados, tote pousa no tampo do trolley z=0.908,
estável). É o mesmo mecanismo do `_build_articulated_object`, logo a mobília
funcionará por esse caminho.

### 1.7 Organização de cena do IH_basic no frame SIMPLE
`extract_ih_layout.py` agora emite um bloco `simple_frame` em `ih_layout.json`:
offsets de tudo relativos ao **TableTrolley** (bancada de montagem = referência).
Os eixos do IH já batem com a convenção industrial do SIMPLE (x = profundidade
robô→bancada, y = lateral), então é só translação — a arrumação relativa do
`.usda` é preservada.

| Elemento | offset (x,y) m | yaw | z |
|---|---|---|---|
| TableTrolley (bancada) | (0, 0) | 0° | — |
| GravityShelf (prateleira) | (+0.397, +1.837) | −127° | — |
| MobileShelvingCart (carrinho) | (+0.852, −2.049) | +54° | — |
| engrenagens ×3 | x[0.13,0.23] y[−0.21,−0.13] | — | 0.849 |
| conectores-T ×4 | x[0.19,0.24] y[0.01,0.18] | — | 0.866 |
| parafusos ×6 | x[0.06,0.18] y[0.43,0.96] | — | ~0.87 |
| bin_b04 ×3 (despacho) | no carrinho, y[−1.9,−0.5] | — | 0.85–1.28 |

As peças ocupam **zonas laterais distintas** na bancada (engrenagens ao centro-frente,
conectores ao centro, parafusos a um lado) — a estrutura espacial natural de um
sortimento. Correção aplicada no parser: prims com `rel material:binding` antes
dos xformOps (as engrenagens) tinham translate lido como (0,0,0); o corte de head
em `\brel\b` foi removido.

### 1.8 Task `G1IndustrialSortingTeleop` — escrita + validada (lado MuJoCo)
`src/simple/tasks/g1_industrial_sorting_teleop.py`, registrada como
`g1_industrial_sorting_teleop` / env `simple/G1IndustrialSortingTeleop-v0`.

- **Mobília:** adicionada no `reset()` da task (após `super().reset()`), como
  `ArticulatedObjectActor` de 0 juntas com chave `furniture_*` (que o spatial DR
  ignora), pose = `table.pose + offset` do `simple_frame`, z=0. Descoberta: o
  layout é montado em `Task.reset` ([core/task.py:155](src/simple/core/task.py#L155)),
  **não** no `DRManager.random_layout` (método morto/bugado — `spatial.apply` não
  existe). Por isso a mobília entra no `reset()`, não numa subclasse de manager.
- **Peças (multi-peça):** target = `industrial_parts:factory_gear_large`;
  distractors = 3 peças **distintas** do subconjunto industrial do graspnet1b
  (broca, chaves, workpieces, parafuso, cadeado, fita). Distintas de propósito: o
  motor nomeia o corpo por `asset.label`, e labels duplicados colidem no compile
  (confirmado) — por isso `allow_duplicates=False`.
- **Recompensa multi-peça:** `compute_reward` = fração das peças (target +
  distractors) dentro do tote (proximidade XY + contato geom↔geom no `mjData`).
- **Alinhamento Z:** `table_height = 0.854` (tampo do trolley) → topo da table box
  coincide com o tampo do mesh; base da mobília em z=0 (piso; g1_sonic usa
  `z_minus=0`).

**Fix de robustez no motor** ([engines/mujoco.py:242](src/simple/engines/mujoco.py#L242)):
a init de juntas articuladas assumia uma chave exata `"articulated"`; agora é
guardada por `if self.articulated_object_joints and "articulated" in actors` —
a mobília (articulated de 0 juntas, chave `furniture_*`) não dispara mais o
`KeyError`. Não afeta tasks articuladas reais (têm a chave + juntas) nem as sem
articulação (lista vazia/None → pula).

**Validação end-to-end (lado MuJoCo)** via `scripts/industrial/validate_sorting_task.py`
(cirúrgica) **e** `gym.make("simple/G1IndustrialSortingTeleop-v0") + env.reset()`
com o `SonicLocoManipEnv` real: cena compila (nbody=54, ngeom=471) com o robô, as
**3 mobílias** como corpos de colisão reais, o target (gear), os distractors e o
tote (bin_b04). As peças pousam na bancada. **Nenhum Plano B acionado.**

Pendente (precisa da stack WBC/hardware, fora do offline): render fotorrealista no
Isaac Sim (a mobília via caminho articulated no engine do Isaac) e a teleoperação
humana de fato (VR/WBC) fechando o loop dual-sim com `--record`.

### 1.9 Refatoração v2 — sorting com totes coloridos (pós-primeira teleop)
Primeira teleop rodou com sucesso; refatoração pedida em cima dela. Task agora é
**"parafusos → tote VERMELHO → prateleira ESQUERDA; chaves de fenda → tote AZUL →
prateleira DIREITA"** (`version: 2.0`).

**Performance — GravityShelf removida.** A malha visual dela sozinha tem 213k
verts / 392k faces (+48 cascos), contra 23k verts por carrinho. No lugar dela
entrou uma **segunda instância** do MobileShelvingCart. Como o engine anexa
mobília sem prefixo de nome (nomes duplicados não compilam), o novo
`scripts/industrial/make_shelf_variants.py` gera
`MobileShelvingCart_C05_left/right_attach.xml` renomeando só os identificadores
(`model=`/`name=`/`mesh=`) e **compartilhando os mesmos OBJs no disco** (zero
duplicação de asset). Prateleira esquerda no ponto IH original do carrinho
(−y), direita no ponto da GravityShelf removida (+y), yaw espelhado.

**Totes coloridos (visíveis no MuJoCo).** `totes.py` ganhou variantes aditivas
`bin_b04_red`/`bin_b04_blue`: mesmas malhas/USD do `bin_b04`, mas `uid==label`
distintos (dois totes na mesma cena — o engine nomeia corpos por `asset.label`)
e um campo `rgba` novo no `TotesAsset`. O engine (`_build_object`) passou a ler
`asset.rgba` quando presente (default continua o branco histórico — 1 linha).
Variantes ficam fora do `sample()` aleatório (só por nome explícito).
**Nota:** o tint é lado-MuJoCo; o replay no Isaac renderiza o material do USD
do bin_b04 (override de cor lá é etapa do render, pendente).

**Cena e robô.**
- Robô mudou de lado: spawn em **+x** (`robot_region` x≈0.64) com **yaw 180°**
  via `robot_orientation_region` (quat `[0,0,0,1]`) — de frente para a bancada.
  Da nova perspectiva: prateleira esquerda em −y, direita em +y; o tote vermelho
  nasce à esquerda do robô e o azul à direita (mnemônico consistente).
- Peças na bancada: target = `graspnet1b:27` (metal screw) + distractors
  forçados a serem as duas chaves de fenda (`19`,`20` via `exclude`). Regiões
  espelhadas para a banda alcançável do lado +x da mesa.
- Totes adicionados no `reset()` (chaves `tote_*`, ignoradas pelo spatial DR)
  em posições fixas na bancada, afastadas das regiões de spawn das peças para
  peça nunca nascer embaixo de tote.

**Recompensa / sucesso (multi-condição).** 0.25 por condição, checadas em
estado **vivo** do MuJoCo (xpos + contatos, não poses iniciais do layout):
`[≥1 parafuso no tote vermelho] + [vermelho apoiado na prateleira ESQUERDA] +
[≥1 chave no tote azul] + [azul apoiado na prateleira DIREITA]`. Com
`success_criteria=0.9`, o episódio só completa com as 4 (reward 1.0).
Cuidado de matching: "screwdriver" contém "screw" — a classificação testa
screwdriver primeiro.

**Validação** (`validate_sorting_task.py`, reescrito, 4 estágios):
1. import/registro; 2. layout stub (poses da mobília/totes + rgba);
3. env real via `gym.make`+`reset` (nbody=54: bench + 2 prateleiras, **sem**
GravityShelf; totes com rgba correto nos geoms; screw + 2 screwdrivers; pélvis
em +x com eixo-x = (−1,0,0)); 4. **lógica da recompensa determinística**: corpos
posados cinematicamente (~2mm de penetração + `mj_forward`, sem settling
dinâmico — settling com `mj_step` cru é flaky porque o robô sem WBC desaba e
perturba a cena) → goal = **1.0**, prateleiras trocadas = **0.5** (crédito de
conteúdo sem crédito de posicionamento). PASS 2× consecutivas.

Fixes de percurso: `totes.py` procurava `{variant}.usd` em vez do
`{base}.usd` compartilhado; snapshots numpy do validador precisavam de
`.copy()` (views vivas do `mjData` mudavam durante o estágio 4).

### 1.10 v2.1 — DR de quantidade de peças + fix do giro de 180° na estabilização

**Bug do giro de 180° (diagnóstico confirmado).** O robô spawna corretamente com
yaw=π (o engine anexa o MJCF na pose do layout — validado), mas girava
ativamente de volta para 0° na estabilização. Causa: a política de pernas
([g1_gear_wbc_policy.py](third_party/decoupled_wbc/control/policy/g1_gear_wbc_policy.py))
faz **servo de yaw para um alvo ABSOLUTO no mundo** (`navigate_cmd[3]`;
`yaw_error = target_yaw − current_yaw`, vyaw saturado em ±1 rad/s), e havia
**duas fontes** mandando 0 absoluto:
1. Estabilização: `get_stabilize_action` (agents vuer/pico) enviava
   `DEFAULT_NAV_CMD = [0,0,0,0]` → target_yaw = 0.
2. Teleop ativo: `VuerStreamer` inicializava `target_yaw = 0.0` e integrava o
   thumbstick a partir daí → mesmo consertando (1), o robô giraria ao ativar.

**Fix (todo no nosso código, nada no third_party):**
- `SonicWbcAgent._spawn_yaw()` (novo, base compartilhada): yaw absoluto do
  `robot.spawn_pose` (quat wxyz → atan2). Matemática validada (π, 0, π/4).
- `vuer_decoupled_agent.get_stabilize_action` e `pico_decoupled_agent`:
  `nav_cmd[3] = _spawn_yaw()` no goal de estabilização.
- `VuerStreamer.reset_status(initial_yaw)` + `reset_policy` semeia com o spawn
  yaw a cada episódio. (PicoStreamer ainda integra de 0 — anotado no código;
  fluxo pico não é o usado.)
- Validação offline: matemática + assinatura + imports ✓. A confirmação
  comportamental final é na teleop real (o loop WBC não roda offline).

**DR de quantidade de peças (1..4 por classe).** Cada episódio sorteia
`n_screws, n_drivers ~ U{1..4}` (`_spawn_parts`):
- O target do spatial DR é sempre o parafuso #1; os demais são **cópias
  rotuladas** de assets graspnet (`metal_screw_2..4`; chaves alternando
  `blue_screwdriver`/`red_screwdriver`/`blue_screwdriver_2`/`red_screwdriver_2`)
  — labels únicos porque o engine nomeia corpos por `asset.label` (o `load()`
  do graspnet cria objeto novo por chamada, mutação segura).
- `DistractorDRCfg` foi removido (o pipeline de distractors não spawna assets
  repetidos — o dict é chaveado por id).
- Posições: rejection-sampling na banda alcançável (`x[0.12,0.28]`,
  `y[−0.20,0.58]`), separação ≥0.10 entre peças/target e ≥0.28 dos totes;
  orientação = stable pose + yaw aleatório (receita do spatial DR).
- **Replay-safe:** a lista de spawns vai em `state_dict["extra_parts"]`; no
  reset com `options["state_dict"]`, as mesmas instâncias/poses são recriadas
  (nomes de corpo têm que bater com a gravação). Reprodutibilidade vem do
  state_dict (o pipeline DR usa `random` global não-semeado, mesmo padrão).
- A recompensa multi-peça já iterava sobre labels — nada mudou na regra
  (≥1 de cada classe no tote certo + totes nas prateleiras certas).

**Validação** (`validate_sorting_task.py`, estágios novos 3b): contagens em
3 resets = (3,2), (3,4), (3,1) ∈ [1,4]; corpos únicos presentes na cena
compilada; separação ok; **roundtrip de replay** (reset com o `state_dict`
gravado → chaves de peças idênticas) ✓; recompensa cinemática segue 1.0/0.5.
`version: 2.1`. FULL VALIDATION: PASS.

### 1.11 Reversão do spawn 180° — teleop não respondia ao VR

Ao teleoperar a v2.1 com o robô girado (yaw=π), **o robô não respondia aos
comandos do VR**. Investigação + decisão do usuário: reverter o spawn ao formato
canônico e **arranjar a cena em torno do robô**, não o robô em torno da cena.

**Causa.** Spawnar o robô numa orientação **não-canônica** (yaw=π). O stack de
teleop (política WBC de pernas + retargeting de mãos) é calibrado para o heading
canônico (~0): a política de pernas serve o yaw da base para um alvo **absoluto
no mundo** (`navigate_cmd[3]`), e o frame de retargeting das mãos assume o
heading da base — com a base em π, os alvos de pulso saem torcidos/inalcançáveis
e os braços não seguem. As mesmas 6 colisões robô↔bancada existentes (punhos/mãos
tocando o tampo a 0.854 m, 5–8 mm) **já ocorriam na v1 que teleoperou com
sucesso**, então não são o bloqueio — o bloqueio era o yaw.

**Revertido (tudo cirúrgico, sem `git checkout` — os 4 arquivos já tinham
modificações anteriores à sessão):**
- `sonic_wbc_agent.py` (`_spawn_yaw` removido), `vuer_decoupled_agent.py` e
  `pico_decoupled_agent.py` (goals de estabilização de volta a `DEFAULT_NAV_CMD`),
  `vuer_streamer.py` (`reset_status()` sem `initial_yaw`, `target_yaw=0.0`).
  Confirmado: 0 traços de spawn-yaw nos diffs; imports OK. A teleop volta ao
  comportamento exato da v1 (que funcionava).
- Task: `robot_region` de volta ao canônico (`x∈[-0.65,-0.63]`, `y=0.25`),
  `robot_orientation_region` **removido** (yaw 0, de frente para +x).

**Cena rearranjada em torno do robô** (robô em -x olhando +x → esquerda=+y,
direita=-y):
- Prateleira ESQUERDA (destino do tote vermelho) → +y `offset (0.397, 1.837)`;
  prateleira DIREITA (tote azul) → -y `offset (0.852, -2.049)` (lados trocados).
- Tote vermelho → esquerda `offset (-0.18, 0.50)`; tote azul → direita
  `offset (-0.18, -0.55)`.
- Região de spawn das peças → metade -x (alcançável), `x(-0.28,-0.10)
  y(0.05,0.45)`, entre os totes. `target_region` idem (`x[-0.22,-0.15]`).

**Validação:** FULL PASS — robô em -x com eixo-x=(1,0,0) (yaw 0); prateleiras/totes
nos lados corretos; DR de quantidade (2,3)/(2,1)/(3,1); replay ✓; recompensa
goal 1.0 / swap 0.5. Confirmação comportamental final da teleop é na próxima
execução real (loop WBC/VR não roda offline).

### 1.12 Posição de operador via transform rígido (robô fixo, cena mapeada)

O usuário quer a composição de **operador** (robô no lado +x de frente para a
bancada), mas o robô não pode ser movido (yaw≠0 quebra o VR). Exigência precisa:
**a posição E orientação relativas robô↔mobília na posição de operador têm que
ficar idênticas** no novo arranjo.

**Solução (transformação rígida).** A cena é **autorada no frame do operador**
(robô-operador em `(0.84, 0.25)`, yaw π) e mapeada rigidamente para o spawn
canônico (`(-0.64, 0.25)`, yaw 0). O mapa que leva a pose do robô-operador na
canônica é uma **rotação planar de 180° em torno do ponto médio** das duas
posições do robô: `p → S − p` (S = op_xy + canon_xy = `(0.20, 0.50)`),
`yaw → yaw + π`. Aplicado a **tudo** (bancada, 2 prateleiras, 2 totes, região de
peças, região do alvo), preserva **exatamente** todos os transforms
robô→mobília. Como robô-op yaw π e robô-canônico yaw 0 diferem de π, o mapa é
`p → S − p` (reflexão pelo ponto médio).

Implementação (`g1_industrial_sorting_teleop.py`): helpers `_op_to_canon_xy`
(`p→S−p`), `_op_to_canon_yaw` (`wrap(yaw+π)`), `_op_to_canon_region` (AABB→AABB).
`_FURNITURE`/`_TOTES` guardam `op_xy`/`op_yaw` (coords do operador);
`_add_furniture`/`_add_totes` aplicam o mapa na colocação. `table_position`,
`target_region` e `_PART_REGION` também mapeados. O robô (`robot_region`) **não
muda** — fica canônico. A bancada mapeia para `(0.20, 0.25)` e a **malha** do
trolley é yawed π (a caixa `table` é simétrica). Resultado: robô canônico (VR
funciona) vendo a cena como se estivesse do lado do operador.

**Prova (validador, estágio 2c):** para cada mobília, o transform relativo
robô→mobília é calculado no arranjo de operador **e** no mapeado, e comparado —
`rel_op == rel_canon` para bancada `(0.84, 0, −π)`, prateleira esq
`(−0.012, 2.049, −2.2)`, prateleira dir `(0.443, −1.837, 2.2)`. **Idêntico** a
1e-9. `[2c] ... identical: True`. Resto do FULL PASS mantido (robô canônico,
cores, DR de quantidade, replay, recompensa goal 1.0 / swap 0.5).

---

## 2. Como reproduzir

```bash
# 1) Layout a partir do USD do Isaac
.venv/bin/python scripts/industrial/extract_ih_layout.py \
    --usda ref_map/IH_basic.usda --out data/assets/industrial/ih_layout.json

# 2) Extrair uma mobília (baixa do S3 se não houver USD local)
.venv/bin/python scripts/industrial/extract_furniture_mesh.py \
    --url "<url do ih_layout.json>" --name TableTrolley_B02_01 \
    --out-dir data/assets/industrial --max-hulls 32

# 3) Validar a colisão da mobília no MuJoCo
.venv/bin/python scripts/industrial/validate_furniture_mujoco.py \
    --names TableTrolley_B02_01 GravityShelfBinOrganizer_A02_01 MobileShelvingCart_C05_01

# 4) Peças dinâmicas (auto-escala pelo metersPerUnit do USD)
.venv/bin/python scripts/industrial/extract_part_mesh.py \
    --url "<url gear>" --name factory_gear_large --max-hulls 16
.venv/bin/python scripts/industrial/extract_part_mesh.py \
    --url "<url t_connector>" --name t_connector_physics --max-hulls 16

# 5) Validar as peças via AssetManager + drop dinâmico
.venv/bin/python scripts/industrial/validate_parts_mujoco.py

# 6) Validar a task (cena MuJoCo com mobília + peças + tote)
.venv/bin/python scripts/industrial/validate_sorting_task.py
```

Notas técnicas:
- **Unidades:** furniture NVIDIA é autorada em cm (`metersPerUnit=0.01`); o
  `IH_basic.usda` corrige com `unitsResolve=(0.01,…)`. O extrator baca esse ×0.01
  → OBJ em metros, batendo com MuJoCo (`metersPerUnit=1`, up=Z).
- **Colisão vs visual:** cada mobília entra como **corpo estático** — malha visual
  completa (`class="industrial_vis"`, `contype=0`) + N cascos convexos
  (`class="industrial_col"`). Formato idêntico ao contrato que `assets/totes.py`
  já usa (lista de OBJs convexos por asset).
- **Sincronia dual-sim:** o `uid`/`label` de cada asset deve ser idêntico nos dois
  motores (bug histórico 2.5.1/4.1 do `INDUSTRIAL_ADAPTATION_CHANGES.md`). Um único
  ponto de nome no futuro `AssetManager` evita isso.

---

## 3. Pendências (restante do Plano A, ainda não feito)

Ordem sugerida:

1. ~~**Peças pegáveis** (`factory_gear_large`, `t_connector_physics`)~~ — **FEITO**
   (ver 1.5). Corpos dinâmicos com cascos convexos + `stable_poses`, validados.
2. ~~**`IndustrialPartsManager`**~~ — **FEITO** (ver 1.5). Falta ainda decidir se a
   **mobília estática** entra num manager próprio ou é incluída direto na cena da
   task via os `*.mjcf.xml` gerados (mais provável, já que furniture não é um
   "objeto pegável" e sim prop de cena).
   - **`bin_b02` (pendente):** é um USD-wrapper SimReady multi-arquivo (referencia
     `bin_b02_base.usd` + payloads), então precisa da **árvore completa** de
     payloads baixada — não é single-file como gear/t_connector. Análogo ao
     `bin_b04` que já existe pronto em `src/simple/assets/totes/bin_b04/`. Enquanto
     isso, o tote `bin_b04` (já funcional) serve de container. Adicionar `bin_b02`
     depois, ou ao `totes` (é um tote) ou ao `industrial_parts`.
3. **Task `G1IndustrialSortingTeleop`** (`src/simple/tasks/`): DR derivada do
   `ih_layout.json`, mobília como props de colisão reais, `obj_surface_map`
   apontando para tampo do trolley / níveis do carrinho / prateleira.
4. **Lógica multi-peça:** `check_object_in_container`/`compute_reward` iterando
   sobre N alvos; definir a regra de sortimento (cada tipo de peça no seu bin, ou
   todas num tote). `check_container_on_<mobília>` por contato geom↔geom.
5. **Registro** em `tasks/__init__.py` + env ID em `envs/__init__.py` (só adições).
6. **Validação dual-sim real:** `scripts/test_env.py` / `preview_teleop_env.py`
   → `teleop_decoupled_wbc.py` (grava MuJoCo) → `render_decoupled_wbc.py --record`
   (replay Isaac Sim).

**Restrição em vigor:** seguir só o Plano A. Se a colisão de malha de alguma peça
dinâmica (ou o custo em cena cheia) ficar ruim, **parar e pedir autorização**
antes de cair para o Plano B (colisão por primitivos).
