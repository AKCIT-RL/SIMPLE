# Playbook — Converter um ambiente do Isaac Sim em task do SIMPLE

Guia operacional para um agente executar a conversão completa: recebe um
ambiente montado no Isaac (`.usda` + USDs dos componentes) e uma descrição de
tarefa, e entrega uma task funcional no pipeline do SIMPLE — **teleoperação no
MuJoCo + renderização no Isaac Sim**.

Destilado da conversão real de `ref_map/IH_basic.usda` → `G1IndustrialSortingTeleop`.
Referência de resultado: [`INDUSTRIAL_IH_PIPELINE.md`](./INDUSTRIAL_IH_PIPELINE.md).
Cada armadilha listada aqui **custou uma sessão de depuração** — leia a §8 antes
de começar, não depois.

---

## 0. Regras que economizam horas

1. **Você provavelmente consegue rodar o Isaac Sim.** O ambiente de execução
   costuma ser a própria máquina do usuário, com GPU. Boota em ~90 s.
   **Nunca diga que não dá para verificar a renderização** sem tentar. Renderize
   e **meça pixels**; não confie em inspeção visual nem em suposição.
2. **Meça, não deduza.** Todo diagnóstico deste projeto que veio de raciocínio
   puro estava errado; todos os que vieram de medição estavam certos.
3. **Um artefato correto ≠ render correto.** Um USD pode estar perfeitamente
   composto (verificável com `pxr`) e mesmo assim renderizar errado. Valide nos
   dois níveis.
4. **Não mexa no robô.** A stack WBC é calibrada para o spawn canônico. Mover ou
   girar o robô quebra a teleoperação. **Adapte a cena ao robô.**
5. **Mudanças no engine devem ser aditivas.** `engines/*.py` é compartilhado por
   dezenas de tasks: adicione flags opcionais (`getattr(asset, "x", default)`),
   nunca altere o comportamento existente.

---

## 1. Entradas a pedir ao usuário

- O `.usda` do ambiente e os USDs/URLs dos componentes.
- A **descrição da tarefa**: objetivo, objetos manipulados, destinos, e o que
  conta como sucesso.
- Decisões que mudam o projeto (pergunte, não presuma):
  - mobília com **colisão de malha real** ou aproximada por primitivos?
  - cenário de render: o USD do usuário ou um cenário genérico já integrado?
  - assets sem equivalente no MuJoCo: substituir ou criar `AssetManager` novo?

---

## 2. Fase 1 — Extrair o layout

```bash
.venv/bin/python scripts/industrial/extract_ih_layout.py \
    --usda <ambiente>.usda --out data/assets/<proj>/layout.json
```

Produz poses, yaws (quaternion→Z), classificação dos prims e um bloco
`simple_frame` com offsets relativos.

- `metersPerUnit` e `unitsResolve` **importam** — anote-os, serão necessários nas
  duas fases seguintes.
- Prims com `rel material:binding` **antes** dos xformOps quebram parsers
  ingênuos (foi um bug real: engrenagens liam translate `(0,0,0)`).
- O JSON é **derivação**, não dependência de runtime: as poses finais vão
  hardcoded na task.

---

## 3. Fase 2 — Converter assets para o MuJoCo

`USD → OBJ (tesselado, em metros) → VHACD → MJCF`, via
`scripts/industrial/extract_furniture_mesh.py` (estático) e
`extract_part_mesh.py` (dinâmico). Ambos usam `_usd_common.py`, que carrega o
`pxr` **sem** subir o Kit.

**Escala é o erro nº 1.** Ative `--auto-units` ou confira `metersPerUnit`:
mobília de armazém da NVIDIA é em **centímetros**; props do Isaac (Props/Factory)
são em **metros**. Errar aqui dá objetos 100× maiores.

**Estático vs dinâmico:**
| | Estático (mobília) | Dinâmico (peças) |
|---|---|---|
| MuJoCo | MJCF full-model + `attach` | cascos convexos + `freejoint` (o engine monta) |
| Precisa | `_attach.xml` | `convex_piece_*.obj` + `stable_poses.npy` |
| Convenção de `stable_poses` | — | `z = -min_z` após a orientação estável (ver `dr/spatial.py`) |

**Múltiplas instâncias do mesmo asset:** o engine anexa MJCF **sem prefixo de
nome** → nomes duplicados não compilam. Gere variantes renomeando apenas os
identificadores e **compartilhando as malhas** (ver `make_shelf_variants.py`).

**Valide cada asset isoladamente** antes de montar a cena
(`validate_furniture_mujoco.py`, `validate_parts_mujoco.py`): sem tunelamento,
contatos limitados, velocidade finita, repouso estável.

**Versione as malhas convertidas.** `data/` costuma ser gitignored, mas obrigar
cada usuário a reconverter é atrito desnecessário. Comite via **Git LFS** apenas
o que o runtime precisa — `*_attach.xml` + OBJs visuais e de colisão (no nosso
caso, 6.7 MB). **Não** comite os USDs baixados (só servem para regenerar; o
Isaac referencia os originais remotamente) nem assets que a task deixou de usar.
Exige negações no `.gitignore` — teste cada caminho com `git check-ignore` e
confirme o filtro com `git check-attr filter <arquivo>`.

---

## 4. Fase 3 — Escrever a task

Copie a task existente mais próxima (`src/simple/tasks/`) e ajuste. Pontos que
**não são óbvios**:

**O layout é montado em `Task.reset`**, não no `DRManager.random_layout` (esse
método é código morto e bugado). Tudo que a task adiciona vai **depois** do
`super().reset()`.

**Consequência crítica:** os passes de DR **já rodaram** nesse ponto. Todo
`ObjectActor` adicionado depois precisa de `actor.set_material(...)` — senão o
**render** morre com `'ObjectActor' object has no attribute 'material'`
(o MuJoCo não lê esse campo, então o bug só aparece no Isaac).

**Chaves de ator** decidem quem o spatial DR posiciona: ele trata
`target`, `container`, `articulated`, `robot` e `*distractor*`. Qualquer outra
chave (`furniture_*`, `tote_*`) é **ignorada** — use isso para poses fixas.

**Mobília estática** entra como `ArticulatedObjectActor` de **0 juntas**
(`MjSpec.from_file` + `attach` no MuJoCo). No Isaac ela precisa de um caminho de
**prop estático** — `SingleArticulation` exige articulação real.

**Robô fixo, cena móvel.** Se a composição desejada exigir o robô em outra pose,
autore a cena nesse frame e **mapeie rigidamente** para o spawn canônico
(rotação de 180° em torno do ponto médio das duas posições do robô, no nosso
caso). Isso preserva a geometria relativa robô↔cena exatamente — e você deve
**provar** isso num teste.

**Recompensa** sobre estado vivo do MuJoCo (`mjData`: `xpos` + contatos
geom↔geom), nunca sobre as poses iniciais do layout. Corpos são nomeados por
`asset.label` → **labels duplicados não compilam**; instâncias repetidas exigem
cópias com label único.

**Replay:** tudo que a task sorteia (quantidades, spawns, prompt) precisa ir no
`state_dict` e ser restaurado quando `options["state_dict"]` existir, ou o replay
diverge da gravação.

---

## 5. Fase 4 — Renderização no Isaac

Só comece depois que a teleop no MuJoCo estiver estável.

- **Materiais MDL são relativos à origem do asset.** Um `.usd` baixado solto
  perde as texturas (`could not find module ...Materials::Base...`). Referencie a
  **URL remota original** e passe URLs (`http/https/omniverse`) **sem**
  `resolve_data_path`/`abspath`.
- **Unidades de novo:** `add_reference_to_stage` **não** converte. Aplique
  `set_local_scale(metersPerUnit)` no prim.
- **Cor de objeto:** editar o material do próprio asset **não funciona**. Crie um
  material OmniPBR e **vincule** ao mesh (`strongerThanDescendants`) — e faça isso
  **no fim do bloco de reset**, com o stage inicializado. Nem em
  `__create_object`, nem logo após `world.reset()`.
- **De-instancie antes de vincular:** autorar em *instance proxy* é ilegal em USD
  e **segfaulta**. `SetInstanceable(False)` no subtree primeiro.

---

## 6. Validação — o que exigir de si mesmo

Escreva um `validate_<task>.py` que rode **offline** (sem VR, sem Isaac) e cubra:

1. import + registro do env;
2. poses de cena; **prove** invariantes geométricos que você prometeu;
3. cena real via `gym.make` + `reset`: corpos presentes, pose do robô, DR nos
   intervalos, *roundtrip* de replay;
4. recompensa: caso-objetivo, casos negativos e limiares — de forma
   **determinística** (pose cinemática + `mj_forward`; **não** deixe assentar com
   `mj_step` cru: o robô sem controlador desaba e contamina a medição).

Invariantes que valem um teste dedicado (cada um pegou um bug real):
- todo `ObjectActor` tem `.material`;
- objetos "escondidos" não contam para prompt nem recompensa;
- quantidades pedidas ≤ disponíveis.

Para a renderização: renderize e **meça pixels**. Salve o PNG e olhe.

---

## 7. Documentação a entregar

- Resumo do pipeline (o quê, como rodar, decisões).
- Guia do usuário: **pré-requisitos e quais assets precisam ser gerados**
  (o `data/` costuma ser gitignored — o usuário terá de regenerar).
- Registro das armadilhas com **causa real**, não só o sintoma.

---

## 8. Armadilhas conhecidas (leia antes de codar)

| Sintoma | Causa real | Onde |
|---|---|---|
| Robô gira sozinho ao estabilizar | `navigate_cmd[3]` é yaw **absoluto**; o default manda 0 | agentes WBC |
| Robô não responde ao VR | Robô fora do spawn canônico | task |
| "Reset em loop" | Combo de reset colide com o gesto de agarrar (squeeze) | binding do VR |
| Comportamento inconsistente entre episódios | Objetos de padding abaixo do piso **colidindo** com o plano infinito e sendo ejetados a ~140 m/s | task |
| Robô manipula objetos que sumiram (só no render) | Poses gravadas resolvidas por `asset.name`, que **repete** entre cópias do mesmo asset; as juntas são `{label}_joint` | render CLI |
| Todos os episódios renderizados com o mesmo prompt | Prompt lido de `tasks.jsonl[0]` e congelado na criação do exporter | render CLI |
| Segfault ao montar cena | Arena do MuJoCo estourou | `mjSpec.memory` |
| Segfault ao colorir objeto | Bind em *instance proxy* | de-instanciar antes |
| Episódio não reseta no sucesso | Rodou sem `--record` | CLI |
| Prompt errado no dataset | `frame["task"]` é rejeitado pelo `validate_frame` | use `exporter.task` |
| `Invalid name 'articulate_base'` | Guarda `is not None` numa lista **vazia** | engine MuJoCo |
| Objeto 100× maior / invisível | `metersPerUnit` não aplicado | extração e Isaac |
| Sem textura no render | MDL relativo à origem remota do asset | usar URL remota |
| Objeto cinza apesar do USD certo | Material do asset não é honrado; e o **momento** do bind importa | Isaac |
| `'ObjectActor' has no attribute 'material'` | Ator adicionado após os passes de DR | task |
| Labels duplicados não compilam | Corpos são nomeados por `asset.label` | task |

---

## 9. Postura esperada

- **Confirme decisões de projeto** antes de implementar; não presuma.
- **Reporte o que falhou** e o que ficou por fazer, sem maquiar.
- **Corrija a causa**, não o sintoma — e diga qual era a causa.
- Ao descobrir que um diagnóstico anterior estava errado, **retrate-se
  explicitamente**; medições ruins geram conclusões ruins que contaminam o resto.
