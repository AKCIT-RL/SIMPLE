# Diagnóstico e correção: qualidade visual da renderização Isaac Sim

O pipeline de renderização Isaac Sim descrito em
`docs/source/tutorials/isaac_warehouse_rendering.md` produz um vídeo de
ponta a ponta para `g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop`,
mas passou por duas rodadas de correção de qualidade visual. Este documento
registra o diagnóstico e o estado atual — o que já foi corrigido e o que
ainda depende de rodar um diagnóstico na máquina com GPU antes de decidir a
correção final.

## Sintomas originais (1ª rodada)

1. Extremamente escuro — o backdrop Warehouse nem aparecia.
2. Basicamente preto-e-branco — sem cor/textura perceptível.
3. As totes distratoras `bin_b04` pareciam não aparecer nas prateleiras.

## O que resolveu a escuridão (sem mudar código)

O caminho do Warehouse estava configurado como `omniverse://localhost/...`
— essa máquina não tem nenhum Nucleus rodando em `localhost`
(`omni.client.list`/`stat` retornam `Result.ERROR_CONNECTION`, confirmado
por diagnóstico). Trocando pela URL de nuvem da NVIDIA:
```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd
```
a própria luz do ambiente Warehouse já ilumina a cena corretamente — **sem
precisar de nenhuma mudança de código**. Uma dome light condicional e um
alcance ampliado de `LightingDRCfg` foram implementados numa primeira
tentativa (antes de confirmar a causa real) e **foram revertidos** depois
de confirmado que não faziam falta — não vale carregar código/config sem
necessidade real.

## O que ainda restava depois de corrigir o Warehouse (2ª rodada)

Com um frame real inspecionado (`media_gen/l1_shelf_isaac_render.png`):

1. `corridor0` e as totes renderizavam com a malha certa, mas sem
   cor/textura — cinza, ou cores default arbitrárias do Kit (esverdeado
   pro `bin_b04`, azulado pro `toteweg`) atribuídas a prims sem material
   vinculado.
2. A prateleira parecia ~10cm alta e à frente demais, dando a impressão de
   totes "no fundo" de cada nível.
3. **Achado maior, não relatado inicialmente pelo usuário mas descoberto
   ao investigar**: só 2 totes apareciam no render (uma `bin_b04` + a
   `toteweg`-alvo), nunca as ~15-18 `bin_b04` simuladas no MuJoCo.

### Causa do achado maior: um só prim Isaac por asset, não por instância

Em `IsaacSimSimulator.update_layout()`/`__create_object`/`__update_object`
(`src/simple/engines/isaacsim.py`), o prim Isaac e a entrada em
`self.objects` eram chaveados só por `object_info.asset.label` — sem
sufixo por instância. Como todas as ~15-18 totes distratoras compartilham
o label `bin_b04`, elas colapsavam num **único** prim
(`/World/workspace/bin_b04`); a cada iteração do loop de objetos,
`__update_object` sobrescrevia a pose desse mesmo prim — só a última
`bin_b04` processada (ordem de inserção do dict `layout.actors`) ficava
visível. Esse bug é anterior a esta investigação toda — só nunca tinha
aparecido porque nenhuma outra task no repo spawna várias instâncias
simultâneas do mesmo asset.

**Corrigido**: `update_layout()` agora calcula, a cada reset, quais labels
se repetem entre os atores da cena (mesmo princípio de
`MujocoSimulator._dup_object_labels`, `engines/mujoco.py`) e usa uma chave
desambiguada (`f"{label}_{obj_name}"` só quando o label se repete; bare
label nos demais casos, preservando o comportamento de toda outra task com
objetos únicos). `__create_object`/`__update_object` passaram a receber e
usar essa chave tanto pro prim path quanto pra `self.objects[...]`.

**Bug irmão corrigido em `MujocoSimulator.get_states()`/`obj_names`**
(`engines/mujoco.py`): a lista `self.obj_names`, consumida por
`IsaacSimSimulator.sync_states()` pra decidir qual prim Isaac mover a cada
passo de física, também usava o label bruto em vez do `body_name`
desambiguado (que já existia calculado ali do lado, só não era usado) —
mesma classe de bug, na sincronização por-frame em vez da criação inicial.
Agora usa `body_name`, consistente com a nova chave do lado Isaac.

### Material ausente em `corridor0`/`toteweg` — confirmado, resolvido

Inspecionando os USD crate binários diretamente: `toteweg.usd` e
`corridor0.usd` têm **zero** ocorrências de `Looks`/`Material`/`Shader`/
`.mdl`/`.png`/`displayColor` — geometria pura, sem material nenhum
embutido. Em contraste, `bin_b04` tem `Looks` + `materials/Plastic_B.mdl` +
texturas de verdade.

**Corrigido**: `__create_object` agora vincula um material
`UsdPreviewSurface` de cor sólida (`_bind_solid_color`, novo método) pra
qualquer asset listado em `IsaacSimSimulator._DEFAULT_MATERIAL_COLORS`
(hoje só `corridor0` e `toteweg`) que não tenha nenhum prim
`Looks/material_*` já — não mexe em `bin_b04` nem em qualquer asset que já
carregue material próprio. Cores de partida (não calibradas contra um
render real, ajustar por inspeção visual):
```python
_DEFAULT_MATERIAL_COLORS = {
    "corridor0": (0.55, 0.55, 0.58),  # cinza metálico
    "toteweg": (0.45, 0.5, 0.42),     # tote industrial neutro (não-alvo)
}
```

### Destaque azul da tote-alvo — portado pro Isaac

`_TARGET_TOTE_RGBA` (`ObjectActor.rgba`) era lido só pelo
`MujocoSimulator._build_object` — o Isaac nunca aplicava, então não dava
pra saber visualmente qual tote é o alvo num render Isaac. **Corrigido**:
`__update_object` agora lê `getattr(obj_info, "rgba", None)` e, se
presente, vincula um material de cor sólida sobrepondo o material base —
a tote-alvo aparece azul no Isaac igual já acontece no MuJoCo.

## Resultado dos diagnósticos (3ª rodada) e correção final

### Piso do Warehouse — confirmado OK, não é a causa

AABB do `warehouse.usd` referenciado sozinho: `min Z ≈ -0.0000687`, ou
seja, o piso já está em z=0. Não precisa de nenhum alinhamento.

### `corridor0` — deslocamento real em X/Y (não em Z)

Comparando o AABB do `corridor0.usd` bruto (referenciado sozinho) contra o
AABB derivado do `MJCF/collision/*.obj` (calculado localmente):

| Eixo | Diferença de largura | Offset de centro |
|---|---|---|
| X | -1.8cm (praticamente igual) | **+18.5cm** |
| Y | **-30cm** (USD bem mais estreito) | **+22cm** |
| Z | -1.7cm (praticamente igual) | ~0cm |

Z está correto — a impressão de "flutuando" era efeito colateral do
deslocamento horizontal (X/Y), não da altura. Y tem, além do deslocamento,
uma diferença real de largura (~30cm) que uma translação simples não
resolve por completo — resíduo aceito, não vale a pena perseguir mais
sem entender por que o USD bruto tem menos extensão em Y do que a malha
de colisão detalhada.

**Corrigido**: `IsaacSimSimulator._VISUAL_POSITION_CORRECTIONS` (novo,
`isaacsim.py`) aplica um offset de `(-0.185, -0.219, 0.0)` só na hora de
posicionar o prim visual do `corridor0` no Isaac, dentro de
`__update_object` — **nunca** na `Pose` real do `StaticObjectActor`
compartilhada com o MuJoCo (mexer nela deslocaria a colisão física
também, desalinhando as totes posicionadas por `SHELF_SPECS`, que são
calibradas contra a colisão real, não contra o USD visual bruto).

### `bin_b04` — material resolve normalmente, não precisa de correção

Diagnóstico (referência isolada do `bin_b04.usd`, sem o resto da cena):
```
/World/bin_b04_debug/bin_b04_inst/Bin_B04_01 Mesh -> material: .../opaque__plastic__bin_b
```
O material vincula e resolve perfeitamente. A aparência sem textura
observada antes era muito provavelmente consequência do bug do prim único
(só uma `bin_b04` de 15-18 chegava a renderizar) — já corrigido. Não foi
aplicada nenhuma mudança adicional para este item; reavaliar visualmente
depois do próximo teste completo.

## Atualização: causa real do deslocamento de `corridor0` — `.usd` obsoleto, não só um offset de origem

Investigação de acompanhamento (ver `media_gen/shelf_swap.png`): o robô passou a ver uma estante tipo
"r" (vista de trás) na frente do spawn, em vez de l1/l3, com totes flutuando. O offset
`_VISUAL_POSITION_CORRECTIONS["corridor0"] = (-0.185, -0.219, 0.0)` documentado acima **tratava o
sintoma, não a causa**: `corridor0.usd` (o que o Isaac de fato renderiza,
`src/simple/assets/fixtures.py`) nunca tinha sido regenerado depois do "Row swap"
(`docs/teleop_simple_study/toteweg_factory_scene_migration_plan.md`, seção 7a), que reescreveu os
`.obj` de colisão/visual do MuJoCo trocando os blocos l1+l3 ↔ r1+r2+r3 de banda-Y. Confirmado por
`md5sum`: o `corridor0.usd` em uso era byte-a-byte idêntico ao `corridor0_legacy/corridor0.usd`
(cópia de rollback pré-swap, mantida de propósito). O offset de 18.5cm/22cm nada mais era do que a
diferença de centro AABB entre o mesh pré-swap (no `.usd`) e o mesh pós-swap (nos `.obj` de colisão
MuJoCo) — coincidindo com o `dx` do próprio Row swap.

**Corrigido**: `third_party/usd2mjcf/eval/regenerate_corridor0_usd.py` (novo) reconstrói
`corridor0.usd`/`corridor0_light.usd` diretamente do `MJCF/visuals/corridor0.obj` atual (pós-swap),
usando `pxr`/`trimesh` já disponíveis em `third_party/usd2mjcf/.venv` — não precisa de GPU nem de Isaac
Sim instalado. `_VISUAL_POSITION_CORRECTIONS["corridor0"]` foi removido de `isaacsim.py`, já que a
origem do `.usd` regenerado agora coincide por construção com a da colisão MuJoCo (verificado via AABB,
delta < 1e-6 m). **Ainda falta confirmar visualmente** rodando o pipeline real numa máquina com GPU —
a checagem de AABB garante alinhamento geométrico, não a aparência final do render.

## Verificação (só possível na máquina com GPU/RTX 4090)

1. Rodar sanity check (sem `--record`), 1 episódio, com a URL de nuvem do
   Warehouse — confirmar visualmente: **todas** as totes `bin_b04`
   aparecem nas prateleiras agora (não só uma), `corridor0`/`toteweg` têm
   cor em vez de cinza/verde/azul default, a tote-alvo aparece destacada
   em azul, e o `corridor0` não parece mais deslocado/flutuando.
2. Se ainda houver desalinhamento visível, ajustar
   `_VISUAL_POSITION_CORRECTIONS["corridor0"]` por inspeção visual (o
   resíduo de ~30cm em Y pode deixar uma borda um pouco diferente mesmo
   depois da correção).
3. Só então rodar com `--record` novamente e conferir o `.mp4` final.
