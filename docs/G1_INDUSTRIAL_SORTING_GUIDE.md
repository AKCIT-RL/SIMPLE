# Guia — Teleoperar e renderizar a task `G1IndustrialSortingTeleop`

Como rodar a task de sortimento industrial: teleoperação com o Meta Quest 3
(física no MuJoCo, com gravação) e renderização fotorrealista no Isaac Sim.

**A tarefa:** o G1 separa peças da bancada em dois totes por cor e leva cada tote
à sua prateleira.

> *"put 2 screwdrivers in the blue tote and 1 screw in the red tote, then place
> the blue tote on the right shelf and the red tote on the left shelf."*

As **quantidades mudam a cada episódio** e aparecem no visor do VR. Visão geral
do pipeline: [`../INDUSTRIAL_IH_PIPELINE.md`](../INDUSTRIAL_IH_PIPELINE.md).

---

## 1. Pré-requisitos

**Hardware**
- GPU NVIDIA compatível com Isaac Sim (necessária para o render; a teleop também
  usa a GPU para as câmeras).
- **Meta Quest 3** (ou headset OpenXR) na **mesma rede Wi-Fi** do PC —
  preferencialmente 5 GHz.

**Software**
- Repositório com o `.venv` do projeto e submódulos (`gear_sonic`,
  `decoupled_wbc`, `televuer`).
- Isaac Sim instalado no `.venv` (é o que o pipeline usa).
- **Acesso à internet na primeira execução:** os assets de mobília são baixados
  do servidor público da NVIDIA, e as texturas da mobília são referenciadas
  remotamente também durante o render.

---

## 2. Você precisa converter os objetos do Isaac?

**Não.** As malhas convertidas da mobília **vêm versionadas** no repositório
(~7 MB, via Git LFS), junto com os totes e os arquivos MJCF. As peças
(parafusos e chaves de fenda, do graspnet) são baixadas automaticamente na
primeira execução.

**A única coisa que você precisa é puxar os arquivos LFS:**

```bash
git lfs install     # uma vez por máquina
git lfs pull
```

> Sem isso, os `.obj` vêm como ponteiros de texto e a task falha ao montar a
> cena. Se ao iniciar você vir erro de malha inválida, quase sempre é LFS não
> puxado.

### Conferir se está tudo no lugar
```bash
.venv/bin/python scripts/industrial/validate_sorting_task.py
```
Deve terminar com `FULL VALIDATION: PASS`. Roda sem VR e sem Isaac — use sempre
que quiser checar o ambiente antes de colocar o headset.

### (Opcional) Regenerar as malhas do zero
Só necessário se quiser reconverter os assets — por exemplo para mudar o número
de cascos de colisão. Baixa da NVIDIA e converte:

```bash
# Bancada (TableTrolley)
.venv/bin/python scripts/industrial/extract_furniture_mesh.py \
  --url "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/TableTrolley_B/TableTrolley_B02_01.usd" \
  --name TableTrolley_B02_01 --out-dir data/assets/industrial --max-hulls 32

# Carrinho de prateleira (base para as duas estantes)
.venv/bin/python scripts/industrial/extract_furniture_mesh.py \
  --url "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Carts/MobileShelvingCart_C/MobileShelvingCart_C05_01.usd" \
  --name MobileShelvingCart_C05_01 --out-dir data/assets/industrial --max-hulls 48

# Variantes esquerda/direita (compartilham as malhas do carrinho)
.venv/bin/python scripts/industrial/make_shelf_variants.py
```

Ao final, rode a validação acima para confirmar que as malhas novas estão boas.

---

## 3. Teleoperar

> **Antes da primeira teleop: certificados SSL.** O stream para o VR usa WebXR
> (HTTPS), então o televuer precisa de um par `cert.pem`/`key.pem`. Eles **não
> vêm no repositório** (são git-ignored) — sem eles a teleop falha no startup
> (arquivo faltando) e, em seguida, cai no crash `Non-positive determinant`.
> Gere uma vez, no local que o televuer procura:
> ```bash
> mkdir -p ~/.config/xr_teleoperate
> openssl req -x509 -newkey rsa:2048 -nodes \
>   -keyout ~/.config/xr_teleoperate/key.pem \
>   -out   ~/.config/xr_teleoperate/cert.pem \
>   -days 3650 -subj "/CN=$(hostname -I | awk '{print $1}')"
> ```
> No **container** (roda como root, `~`=`/root`, e o `.config` é efêmero) gere os
> certs dentro dele ou monte por volume. A ordem de resolução completa (args →
> env vars → `~/.config/xr_teleoperate` → fallback) está em **SSL Certificates
> (televuer)** no `source/tutorials/teleop.md`.

```bash
.venv/bin/python src/simple/cli/teleop_decoupled_wbc.py simple/G1IndustrialSortingTeleop-v0 \
    --sim-mode=mujoco --record --no-headless
```

> **`--record` é obrigatório para o episódio resetar ao concluir a tarefa.** Sem
> ele a detecção de sucesso não roda (ela vive dentro do fluxo de gravação) e
> nada acontece ao completar. Também é o que gera os dados para o render.

**Conectar o headset**
1. O terminal mostra: `Open https://<IP_DO_PC>:8012 in the Meta Quest browser`.
   Na primeira vez o pipeline ONNX do WBC é baixado — acompanhe os logs.
2. Abra no navegador do Quest. **Se o PC tem mais de uma interface de rede** — ou
   o headset alcança o PC por um IP específico, como a sub-rede do robô
   `192.168.123.x` — fixe o WebSocket direto na URL:
   ```
   https://192.168.123.2:8012/?ws=wss://192.168.123.2:8012
   ```
   O `?ws=wss://<IP>:<porta>` diz ao cliente Vuer **exatamente** onde abrir o
   WebSocket, forçando o esquema seguro `wss://` (obrigatório porque a página é
   HTTPS — um `ws://` simples seria bloqueado como *mixed content*). Sem ele, o
   Vuer tenta adivinhar a URL a partir do host da página e pode escolher a
   interface errada.
3. Aceite o aviso de certificado (**Avançado → Prosseguir**) — é autoassinado.
4. Toque em **Enter VR**. Você passa a ver pela câmera do robô.

**Controles**

| Ação | Comando |
|---|---|
| Ativar/desativar os braços | Botão **B** (controle esquerdo) |
| Andar / girar a base | Thumbstick esquerdo / direito |
| Fechar dedos | Gatilho (indicador) e *squeeze* (demais) |
| Descer o robô (elástico) | Clique no thumbstick direito |
| Resetar o episódio | **Squeeze nos dois controles ao mesmo tempo** |

> ⚠️ **Cuidado:** o combo de reset usa o mesmo *squeeze* que fecha as mãos.
> Apertar os dois controles junto — natural ao pegar um tote com as duas mãos —
> **reseta o episódio**. Prefira agarrar com uma mão de cada vez.

**No visor** aparecem o contador de episódios e as quantidades do episódio:
```
SCREWDRIVERS -> BLUE: 2
SCREWS -> RED: 1
```

**Concluir:** coloque **pelo menos** as quantidades pedidas em cada tote, leve o
**vermelho para a prateleira esquerda** e o **azul para a direita**. Ao cumprir
tudo, o episódio salva e reseta sozinho. Objetos a mais não invalidam.

Os dados vão para
`data/teleop_decoupled_wbc/simple/G1IndustrialSortingTeleop-v0/level-0`.

---

## 4. Renderizar

Reproduz os episódios gravados com qualidade fotorrealista. Só as **luzes**
variam entre episódios; texturas e materiais ficam nos originais.

```bash
# teste rápido: 1 episódio, sem gravar
.venv/bin/python src/simple/cli/render_decoupled_wbc.py simple/G1IndustrialSortingTeleop-v0 \
    --data-dir data/teleop_decoupled_wbc/simple/G1IndustrialSortingTeleop-v0/level-0 \
    --num-episodes 1 --no-record

# passada completa, gravando
.venv/bin/python src/simple/cli/render_decoupled_wbc.py simple/G1IndustrialSortingTeleop-v0 \
    --data-dir data/teleop_decoupled_wbc/simple/G1IndustrialSortingTeleop-v0/level-0 \
    --record --save-dir data/render_decoupled_wbc
```

O Isaac leva ~1 min para subir. Renderizar é bem mais lento que tempo real
(~7 frames/s): um episódio longo pode levar dezenas de minutos.

---

## 5. Problemas comuns

| Sintoma | O que fazer |
|---|---|
| Episódio não reseta ao concluir | Faltou `--record` |
| Reseta sozinho ao pegar o tote | *Squeeze* nos dois controles é o atalho de reset — agarre com uma mão por vez |
| Headset não conecta | Mesma rede; aceite o certificado; use 5 GHz |
| Robô "se debate" ao iniciar | Aguarde o pouso do elástico e só então ative com **B** |
| Erro de malha/asset ao iniciar | `git lfs install && git lfs pull` (§2) |
| Mobília sem textura no render | Sem internet: as texturas da mobília são referenciadas do servidor da NVIDIA |
| Material não atualiza no render | `rm -rf ~/.cache/ov/shaders` (recompila sozinho) |

---

## 6. Ajustes rápidos

| O quê | Onde |
|---|---|
| Cor dos totes | `Totes_Variants` em `src/simple/assets/totes.py` (vale para MuJoCo e Isaac) |
| Quantidade máxima de peças | `_MAX_PARTS_PER_CLASS` na task |
| Texto do prompt | `LanguageDRCfg` na task (mantenha os campos `{drivers}`/`{screws}`) |
| Variação de iluminação | `LightingDRCfg` na task |
| Posição de mobília e totes | `_FURNITURE` / `_TOTES` na task (coordenadas no *frame do operador*) |

Task: `src/simple/tasks/g1_industrial_sorting_teleop.py`.
