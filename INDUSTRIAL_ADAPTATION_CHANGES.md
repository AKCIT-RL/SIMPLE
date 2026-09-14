# Adaptação para Ambientes Industriais — Revisão e Correções

Este documento registra o que foi alterado no repositório para suportar as tasks
industriais (`g1_industrial_screw_to_tote_teleop` e `g1_industrial_tote_to_rack_teleop`),
os problemas encontrados na revisão do trabalho original, e as correções aplicadas
para manter o fluxo alinhado com o restante do repositório.

Contexto original: [`handover_prompt_industrial_teleop.md`](./handover_prompt_industrial_teleop.md).

---

## 1. O que a adaptação industrial adicionou

| Arquivo | O que é |
|---|---|
| `src/simple/scenes/warehouse.py` (novo) | `WarehouseSuite` (subclasse de `TabletopScene`) + `WarehouseSceneManager`, registrado como `"warehouse"`. Carrega o USD nativo da NVIDIA (`Isaac/Environments/Simple_Warehouse/warehouse.usd`) via Nucleus. |
| `src/simple/tasks/g1_industrial_screw_to_tote_teleop.py` (novo) | Task: pegar uma peça (graspnet1b) da bancada de montagem, colocar num tote, levar o tote até a bancada de despacho (`table2`). |
| `src/simple/tasks/g1_industrial_tote_to_rack_teleop.py` (novo) | Task: pegar um tote já com peças e colocar sobre um rack/estante industrial (`table2`). |
| `src/simple/scenes/__init__.py`, `src/simple/tasks/__init__.py`, `src/simple/envs/__init__.py` | Exports/registro (`WarehouseSuite`, as duas novas tasks, os dois novos env IDs `simple/G1IndustrialScrewToToteTeleop-v0` e `simple/G1IndustrialToteToRackTeleop-v0`). Só adições, nenhuma linha existente alterada. |
| `src/simple/engines/isaacsim.py` | `__update_scene` ganhou um branch para `WarehouseSuite` (carrega o USD do warehouse em vez do fluxo HSSD). `_setup_table` ganhou fallback de material quando o MDL não resolve. |
| `src/simple/envs/sonic_loco_manip.py` | `sonic_config` virou opcional (`dict | None = None`), para permitir instanciar o env fora do fluxo de teleop completo (ex: via `scripts/test_env.py`). |
| `src/simple/cli/teleop_decoupled_wbc.py` | Guarda `if not record: return` removida de `_on_episode_reset` — o reset do WBC (desliga elastic band, engata política RL) passou a rodar em todo modo, não só gravação. |
| `scripts/test_env.py` | Passa a carregar e repassar `sonic_config` ao `gym.make`, necessário porque as novas tasks usam `SonicLocoManipEnv`. |
| Assets `src/simple/assets/totes/bin_b04/*` | Modelo 3D do tote usado como `container`/`target` nas tasks (mesh, texturas, USD). |

---

## 2. Problemas encontrados na revisão e correções aplicadas

### 2.1 `table2` nunca recebia material — bug pré-existente, não específico do industrial

**O que era.** `MaterialDR.__call__` (`src/simple/dr/material.py`) sempre chamou
`table.set_material(...)`, mas **nunca** `table2.set_material(...)`. Isso já existia
desde o commit inicial do repo (`a73ea6e`, antes de qualquer trabalho industrial) e
afeta **20 tasks nativas** que usam `enable_table2=True` (`pick_between_tables` e
todas as variantes, `sit_mp`, `bend_pick_and_place_on_sofa*`, etc.) — não é um problema
introduzido pela adaptação industrial.

Na prática isso não "quebrava" visualmente porque um prim sem material assume o
preview surface cinza padrão do Omniverse, então passava despercebido.

O primeiro fix aplicado durante a revisão introduziu um campo `fixed_table2_material`
só para `table2`, criando uma assimetria nova: `table` seguia um caminho (`material_mode`),
`table2` seguia outro (override dedicado). Isso foi corrigido na iteração seguinte.

**Correção final.** `MaterialDR` ganhou um único método `_pick_surface_material(fixed_override)`
usado **da mesma forma** para `table` e `table2`:

```python
table_material  = self._pick_surface_material(self.cfg.fixed_table_material)
table2_material = self._pick_surface_material(self.cfg.fixed_table2_material)
```

- Sem override: cai no comportamento de sempre (`material_mode` — random por episódio em
  modo `rand_all`/`rand_tableground`, ou o material de madeira fixo padrão).
- Com override (`fixed_table_material` / `fixed_table2_material`, ambos opcionais e
  simétricos em `MaterialDRCfg`): a mesa correspondente fica travada nesse material.

Efeito colateral (desejado): as 20 tasks nativas com `table2` passam a ter essa mesa
também texturizada via o mecanismo de domain randomization do pipeline base — antes
ela nunca recebia textura nenhuma.

Retrocompatibilidade: o caminho de replay (`load_state_dict`) usa
`ret.get("table2_material")`, então estados gravados antes desta mudança (sem essa
chave) continuam carregando normalmente, sem tentar setar material em `table2`.

### 2.2 Fallback de material em `isaacsim.py` — de "estilo industrial" para neutro

**O que era.** `_setup_table` (`isaacsim.py`), quando `create_mdl_material` falhava ou
não havia material, criava um material sintético OmniPBR "Aço Metálico Escuro
Industrial" (`diffuse=(0.2,0.2,0.2)`, `metallic=0.8`) — e isso rodava para **qualquer**
mesa sem material, de qualquer cena (HSSD incluído), não só warehouse.

**Correção.** Com o fix de 2.1, `table`/`table2` praticamente sempre chegam com um
`mat_info` válido (desde que o randomizer de material rode, o que é o padrão). O
fallback em `isaacsim.py` deixa de ser o caminho "normal" para `table2` e volta a ser
apenas uma rede de segurança genérica (MDL corrompido, falha pontual de Nucleus, etc.),
então trocamos a aparência para algo neutro (`diffuse=(0.5,0.5,0.5)`, `metallic=0.0`,
`roughness=0.5`) em vez de um look industrial específico.

O visual "industrial" (metal) das tasks novas, quando ativado, vem do jeito certo: via
`fixed_table_material`/`fixed_table2_material` no `MaterialDRCfg` de cada task (aponta
para `vMaterials_2/Metal/Metal_Cast.mdl`, confirmado que o identificador existe no
`.mdl`), não de um fallback silencioso de erro. Ver seção 4.2 para como isso é ligado.

### 2.3 Checagem de tipo de cena por string em vez de `isinstance`

**O que era.** `__update_scene` trocou `isinstance(scene, HssdSuite)` /
`isinstance(scene, ShowHouse)` por `type(scene).__name__ == "HssdSuite"` /
`"ShowHouse"` / `"WarehouseSuite"`, apesar de `HssdSuite` já estar importado no topo
do arquivo. Frágil: quebra silenciosamente se qualquer uma dessas classes for
subclassificada no futuro (padrão comum no resto do código).

**Correção.** Revertido para `isinstance`, importando `WarehouseSuite` no topo do
arquivo junto com `HssdSuite` (já existia) e usando `simple.scenes.ShowHouse` (já
importado via `import simple.scenes`).

### 2.4 `except:` mudo ao esconder o ceiling

**O que era.** Um bloco novo em `__update_scene` (visibilidade do ceiling em cenas
HSSD) usava `except: pass`, engolindo qualquer erro sem log — inconsistente com os
outros blocos novos do mesmo arquivo, que sempre imprimem um Warning.

**Correção.** `except Exception as e: print(f"Warning: ...: {e}")`, no mesmo padrão
já usado em outros pontos de `isaacsim.py` (inclusive um bloco pré-existente,
não relacionado a este trabalho, em `add_object` por volta da linha 856).

### 2.5 `WarehouseSuite` com campos declarados e nunca usados

**O que era.** `center_offset` / `center_orientation` foram copiados do padrão de
`HssdSuite`, mas o branch de warehouse em `isaacsim.py` nunca lê esses campos (pula
inteiramente a lógica de `move_surface_to_origin`, que é específica de HSSD).

**Correção.** Campos removidos de `WarehouseSuite`. `TabletopScene` (classe base) não
exige esses atributos — só `table`/`table2`.

### 2.5.1 `TabletopSceneDR.load_state_dict` lia `center_offset`/`center_orientation` sem checar se existiam — bug pré-existente, exposto pelo warehouse

**Sintoma.** Rodar `render_decoupled_wbc.py` com `--record` em
`G1IndustrialScrewToToteTeleop-v0` falhava com `[Replay] Error: 'center_offset'`
durante a etapa de replay no Isaac Sim (renderização fotorrealista dos episódios
gravados no MuJoCo).

**O que era.** Em `src/simple/dr/scene.py`, `TabletopSceneDR.load_state_dict` fazia
`self._inner_state.center_offset = state_dict["center_offset"]` (e o mesmo para
`center_orientation`) incondicionalmente. Esses atributos só existem em `HssdSuite`
(setados em `dr()`/`middle()`) — `WarehouseSuite` e `ShowHouse` nunca os têm, então
`to_dict()` nunca inclui essas chaves para cenas que não são HSSD. Como toda task
nativa usa `scene_manager="hssd"`, esse gap nunca tinha sido exposto antes das tasks
industriais usarem `scene_manager="warehouse"`.

**Correção.** Mesmo padrão já usado para `table2` na mesma função (chave opcional):

```python
if "center_offset" in state_dict:
    self._inner_state.center_offset = state_dict["center_offset"]
if "center_orientation" in state_dict:
    self._inner_state.center_orientation = state_dict["center_orientation"]
```

Não afeta HSSD (a chave sempre existe lá) e resolve o replay para warehouse.

### 2.6 `_on_episode_reset` sem guarda de modo — confirmado intencional

**O que era.** `teleop_decoupled_wbc.py` removeu `if not record: return` de
`_on_episode_reset`, fazendo o reset do WBC (desliga elastic band, engata política RL)
rodar em todo modo, não só gravação.

**Decisão.** Mantido como está. Confirmado com quem fez a mudança: sem essa remoção,
o robô "se debatia" de forma estranha em qualquer sessão sem `--record` ativo; testado
e validado que a remoção da guarda resolve o problema.

### 2.7 Itens verificados e considerados corretos (não mexidos)

- **`dr/scene.py`** (construção de mesa para MuJoCo e Isaac Sim): já era genérico —
  usa `isinstance(self.scene_manager, HssdSceneManager)` e cai num branch `else`
  que usa os valores de `Box` do `TabletopSceneDRCfg` diretamente, sem tocar em
  `scene.conf`. Não foi alterado pela adaptação industrial e não precisou de nenhuma
  mudança: `WarehouseSuite` + `WarehouseSceneManager` já se encaixam nesse branch
  genérico corretamente, porque seguem o mesmo contrato de `TabletopScene` que
  `HssdSuite` segue (subclasse + `set_table`/`set_table2`).
- **`src/simple/engines/mujoco.py`**: não referencia `scene.data_dir`, `scene.conf`
  nem faz `isinstance`/checagem de tipo de cena em nenhum lugar — o motor de física
  (usado para teleoperação, o "sim rápido" da arquitetura Dual-Sim) é agnóstico ao
  tipo de cena. A renderização fotorrealista (HSSD vs. warehouse) é uma
  responsabilidade só do Isaac Sim.
- **Branch HSSD em `isaacsim.py`**: comparado o diff ignorando espaços em branco —
  a lógica ficou byte-a-byte idêntica à original, só foi reindentada para dentro do
  `else` e envolvida em `try/except` de segurança. Nenhum comportamento mudou para
  cenas HSSD existentes.
- **Registro de envs/tasks** (`envs/__init__.py`, `tasks/__init__.py`,
  `scenes/__init__.py`): só adições de linha no fim do arquivo, nenhuma linha
  existente tocada.
- **`sonic_config` opcional em `SonicLocoManipEnv`**: `self.sonic_config = sonic_config or {}`
  só entra em jogo quando `sonic_config` é `None`/falsy; o fluxo principal
  (`teleop_decoupled_wbc.py`) sempre passa um dict real, então não muda nada pra quem
  já usa o CLI de teleop.

---

## 3. Assets de mesa/estante: só existe caixa procedural

Não existe nenhum asset dedicado de mesa ou estante/rack com malha própria em lugar
nenhum do repositório — nem nas tasks nativas nem nas industriais. `table` e `table2`
são sempre `primitive:box` (`src/simple/assets/primitive.py`), texturizado via DR.

O "rack" de `g1_industrial_tote_to_rack_teleop.py` é conceitual: fisicamente é o
mesmo `primitive:box` genérico de qualquer `table2`, distinguido pela instrução em
linguagem natural, pela lógica de sucesso (`check_container_on_rack`) e pelo material
metálico fixo (seção 2.1/2.2). Asset managers existentes no repo, para referência:

| Manager | Conteúdo |
|---|---|
| `primitive` | `Box` — usado para `table`/`table2` em toda task |
| `totes` | bins/caixas (ex: `bin_b04`) |
| `graspnet1b` | objetos pequenos pegáveis |
| `objects` | itens avulsos (ex: `"brasket"`) |
| `articulated` | objetos articulados (portas, forno, torneira, cadeira) |
| `objaverse` | biblioteca genérica de assets 3D |

Um rack com geometria real (prateleiras, estrutura vertical) exigiria um novo
`AssetManager` nos moldes de `totes.py`, com USD/OBJ próprio — não existe hoje.

---

## 4. Correções pós-teste (primeira execução real com `--record`)

Encontradas rodando `render_decoupled_wbc.py` de fato para `G1IndustrialScrewToToteTeleop-v0`.

### 4.1 Alvo (`graspnet1b:63`) não era um parafuso

**Sintoma.** O objeto-alvo renderizado era um frasco de shampoo/condicionador, não um
parafuso, apesar do comentário no código dizer "screw/small part".

**Causa.** `GraspNet_1B_Object_Names[63]` (`src/simple/assets/graspnet.py:85`) é
`"head shoulders supreme"` — confirmado por metadado, sem precisar checagem visual. O
índice correto para parafuso é **27: `"metal screw"`** (`graspnet.py:49`), nunca usado
nas tasks industriais.

**Correção.** Nas duas tasks (`g1_industrial_screw_to_tote_teleop.py`,
`g1_industrial_tote_to_rack_teleop.py`): `target_object` default e
`TargetDRCfg(asset_id=...)` trocados de `"graspnet1b:63"` para `"graspnet1b:27"`; a
lista `exclude` dos distractors trocou o `"63"` manual por `"27"` (mesma finalidade:
evitar sortear o próprio alvo como distractor). Confirmado que os assets do índice 27
(collision mesh, USD, texturas) existem em `data/assets/graspnet/`.

### 4.2 Mesa 1 randomizava, mesa 2 não — assimetria não desejada, agora é tudo-ou-nada

**Sintoma.** Só `table` variava de textura a cada episódio; `table2` ficava sempre em
`Metal_Cast`.

**Causa.** Comportamento configurado de propósito na iteração anterior
(`fixed_table2_material` fixo por padrão nas duas tasks) — não era bug, mas o usuário
apontou que não faz sentido ter só uma mesa fixa e a outra variando: **a configuração
principal deveria ser as duas randomizadas, com uma flag opcional para fixar as duas**
em look metálico industrial juntas.

**Correção.** Removido o `fixed_table2_material` fixo do `MaterialDRCfg` de classe
(agora `material_mode="rand_all"` puro — as duas mesas randomizam de forma independente
por padrão, igual ao resto do pipeline). Adicionado um parâmetro de construtor
`industrial_material: bool = False` nas duas tasks; quando `True`, seta
`fixed_table_material` **e** `fixed_table2_material` juntos para `Metal_Cast` — nunca
só um dos dois:

```python
if industrial_material:
    material_cfg.fixed_table_material = _INDUSTRIAL_METAL_MATERIAL
    material_cfg.fixed_table2_material = _INDUSTRIAL_METAL_MATERIAL
```

Exposto também na CLI de gravação (`teleop_decoupled_wbc.py`, novo flag
`--industrial-material`), já que a escolha de material acontece na gravação em MuJoCo
(`teleop_decoupled_wbc.py`), não no replay/render (`render_decoupled_wbc.py`) — este
último só reproduz o que já foi gravado no `environment_config` do episódio.

Uso:

```bash
# padrão: as duas mesas randomizam (recomendado para treino)
.venv/bin/python src/simple/cli/teleop_decoupled_wbc.py simple/G1IndustrialScrewToToteTeleop-v0 ...

# as duas mesas fixas em metal industrial
.venv/bin/python src/simple/cli/teleop_decoupled_wbc.py simple/G1IndustrialScrewToToteTeleop-v0 --industrial-material ...
```

Nota: `dr_cfgs` é um dict de nível de classe (compartilhado entre instâncias da mesma
task, mesmo padrão pré-existente já usado para `target_object`/`distractor.exclude`) —
`industrial_material=True` fica "grudado" se a mesma classe de task for instanciada de
novo no mesmo processo sem o flag. Isso não muda o comportamento observável no fluxo
normal de uso (um processo = uma sessão = uma task), então não foi alterado.

---

## 5. Pendências (fora do escopo desta revisão)

- **Mismatch de Git LFS** nos assets de `src/simple/assets/totes/bin_b04/`: o commit
  que adicionou o tote gravou os arquivos como blob bruto em vez de ponteiro LFS,
  apesar de `.gitattributes` marcar `*.obj`/`*.usd`/`*.png` para `filter=lfs`. Isso faz
  `git diff`/`git status` mostrarem ~130 arquivos como "modified" sem nenhum conteúdo
  realmente alterado. Combinado deixar para depois.
- `WarehouseSuite.data_dir` (fallback quando `get_assets_root_path()` retorna `None`)
  aponta para um Nucleus local hardcoded (`omniverse://localhost/...`) — só funciona
  se existir um servidor Nucleus local com esse asset. Fora do escopo desta rodada,
  mas vale revisitar se as tasks industriais forem rodar em outra máquina/deploy.
