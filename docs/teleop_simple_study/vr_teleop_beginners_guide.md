# Guia para Iniciantes — Captura de Dados via Teleoperação VR no SIMPLE

Este guia assume que você nunca rodou o repositório antes. Ele cobre, em ordem: setup do
ambiente, download de assets, execução da simulação, conexão com o headset Meta Quest 3,
controle do robô, gravação de episódios e envio dos dados pro Hugging Face.

**Em toda seção, o comando indica de qual diretório ele deve ser executado.** Sempre que
dizemos "raiz do repositório", é a pasta onde estão os arquivos `pyproject.toml`, `Dockerfile`
e `README.md` (ex.: `~/Projects/Teleop/simple_teleoperation/SIMPLE` ou onde você tiver
clonado o projeto).

Se qualquer passo aqui não bater com o que você está vendo na tela, pare e pergunte — não
adivinhe, principalmente na parte de rede/headset e na parte de upload de dados (confirme com o
time qual repositório Hugging Face usar antes de subir algo pela primeira vez).

---

## 1. Pré-requisitos

### 1.1 Hardware/SO do PC

| Componente | Mínimo | Recomendado |
| :--- | :--- | :--- |
| SO | Ubuntu 22.04 | Ubuntu 22.04 |
| CPU | Intel i7 / AMD Ryzen 7 | Intel i9 / AMD Ryzen 9 |
| RAM | 32 GB | 64 GB |
| GPU | NVIDIA RTX 2070 (8GB VRAM) | RTX 3080 Ti / 4090 (16GB+) |
| Driver NVIDIA | 535.x | mais recente |
| CUDA | 12.x | 12.x |
| Python | 3.10 | 3.10 |
| Armazenamento | 50 GB SSD | 100+ GB NVMe |

GPU precisa ser NVIDIA de arquitetura RTX ou mais nova — GTX não é suportada.

### 1.2 Headset

- Meta Quest 3 (é o único headset validado pelo fluxo Vuer/TeleVuer descrito aqui).
- **O headset e o PC precisam estar na mesma rede Wi-Fi** (de preferência 5GHz, ou um
  roteador dedicado só pra isso — redes corporativas com isolamento de cliente costumam
  bloquear a conexão).

---

## 2. Clonar o repositório e trazer os submódulos

Rode de onde você guarda seus projetos (não precisa ser um diretório específico, esse será o
diretório-pai do repo):

```bash
git clone <url_do_repositorio> SIMPLE
cd SIMPLE
```

A partir daqui, **toda a raiz do repositório é `SIMPLE/`** — os comandos abaixo assumem que
você está dentro dela, a menos que eu diga o contrário.

O projeto depende de vários submódulos em `third_party/` (decoupled_wbc, televuer,
gear_sonic, etc.) — se você já tiver clonado sem `--recursive`, busque-os agora:

```bash
# a partir da raiz do repositório (SIMPLE/)
git submodule update --init --recursive
```

---

## 3. Setup do ambiente Python com `uv`

Esta é a via "rápida" oficial do README. Todos os comandos desta seção rodam **a partir da
raiz do repositório**.

### 3.1 Instalar o `uv` (se ainda não tiver)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3.2 Instalar todas as dependências

```bash
# a partir da raiz do repositório (SIMPLE/)
UV_HTTP_TIMEOUT=3000 bash scripts/setup_python_env.sh
```

Esse script roda `uv sync --all-groups`, ou seja, instala **todos** os grupos de
dependências de uma vez — incluindo o grupo `sonic` (que traz `televuer`,
`xrobotoolkit-sdk`, `decoupled-wbc`, necessários pra teleoperação VR), o grupo `lerobot`
(formato de dataset usado na gravação) e o `dev` (que já traz `huggingface_hub[cli]`, usado
mais adiante pra subir dados). Não é preciso instalar nenhum grupo à parte.

Isso cria a pasta `.venv/` na raiz do repositório.

### 3.3 Instalar o CuRobo

```bash
# a partir da raiz do repositório (SIMPLE/)
bash scripts/install_curobo.sh
```

### 3.4 Ativar o ambiente

```bash
# a partir da raiz do repositório (SIMPLE/)
source .venv/bin/activate
```

Você vai precisar rodar esse `source` em todo terminal novo que for usar pra este projeto
(ou usar `uv run <comando>`/`.venv/bin/python3 ...` diretamente sem ativar, como alguns
exemplos abaixo fazem).

### 3.5 Verificar a instalação

```bash
# a partir da raiz do repositório (SIMPLE/), com o venv ativado
python -c "import simple; print(simple.__version__)"
```

Se isso imprimir uma versão sem erro, a base do ambiente está OK.

> Setups alternativos (robo-nix, Docker) existem e estão documentados no `README.md` — este
> guia segue só a via `uv`, que é a mais direta pra começar.

---

## 4. Variáveis de ambiente (`.env`)

Ainda na raiz do repositório, copie o arquivo de exemplo (se ainda não existir um `.env`):

```bash
# a partir da raiz do repositório (SIMPLE/)
cp .env.sample .env
```

Edite `.env` (na raiz do repositório) e confira/ajuste pelo menos:

```
DATA_DIR=<caminho absoluto onde os dados/assets vão ficar>
MUJOCO_GL=egl
```

Você também vai precisar de um **token do Hugging Face** mais adiante (tanto pra baixar
alguns assets quanto pra subir os dados capturados), além do seu nome de operador e dos
repositórios de destino. Adicione ao `.env`:

```
HF_TOKEN=hf_xxx...
HF_REPO_RAW=USC-PSI-Lab/simple-teleop-raw
HF_REPO_RENDERED=USC-PSI-Lab/simple-teleop-rendered
SIMPLE_OPERATOR=<seu-nome>
```

Gere o token em https://huggingface.co/settings/tokens — precisa ter permissão de **escrita**
nos dois repos acima (não só leitura), já que o mesmo token é usado tanto pra baixar assets
quanto pra subir os dados que você captura. `SIMPLE_OPERATOR` é usado para nomear as pastas de
sessão e fica gravado nos metadados de cada captura — combine com o time qual formato de nome
usar (ex.: `joao_silva`).

---

## 5. Baixar os assets necessários

Cenas, materiais e alguns objetos (HSSD, vMaterials, GraspNet) **não ficam no git** — são
baixados de um repositório de dados no Hugging Face. Sem isso, mesmo com o ambiente Python
certo, a simulação vai falhar ao tentar carregar texturas/materiais.

```bash
# a partir da raiz do repositório (SIMPLE/)
bash scripts/pre-minimal-download.sh
```

Isso baixa os arquivos pra dentro de `data/` (na raiz do repositório). Use `--cleanup` no
final se quiser apagar os `.zip` depois de extraídos, pra economizar espaço:

```bash
bash scripts/pre-minimal-download.sh --cleanup
```

> As tasks de `toteweg`/`corridor0` (prateleira → mesa) já têm os próprios assets versionados
> via Git LFS dentro de `src/simple/assets/`, então não dependem deste download — mas as
> demais partes da task (materiais, iluminação) ainda usam o `vMaterials_2` baixado aqui.

---

## 6. Rodar uma pré-visualização sem VR (recomendado antes de ligar o headset)

Antes de mexer com o headset, vale confirmar que a cena carrega e os objetos aparecem
corretamente. Existe um script pra isso:

```bash
# a partir da raiz do repositório (SIMPLE/), com o venv ativado
python src/simple/cli/preview_teleop_env.py <env_id>
```

Troque `<env_id>` pelo id da task, por exemplo:
`simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0`. Veja
`docs/teleop_simple_study/preview_teleop_env.md` (mesma pasta deste guia) pra mais detalhes
sobre esse script.

Os ids de todas as tasks registradas ficam em `src/simple/tasks/*.py` (decorador
`@TaskRegistry.register("...")` — o id do ambiente Gym completo é sempre
`simple/<NomeDaClasse>-v0`, sem o sufixo `Task`).

---

## 7. Conectividade de rede para o Quest 3

1. Confirme que o **PC e o Meta Quest 3 estão na mesma rede Wi-Fi**.
2. Descubra o IP do PC nessa rede:
   ```bash
   ip addr show | grep "inet "
   ```
   (procure o IP da interface conectada ao Wi-Fi/rede local, não o `127.0.0.1`).
3. Se você for rodar dentro do Docker: o `docker-compose.yml` já usa `network_mode: "host"`
   pros serviços de simulação, então as portas ficam expostas diretamente — não precisa
   mapear porta manualmente.

---

## 8. Lançar a teleoperação

O script principal fica em `src/simple/cli/teleop_decoupled_wbc.py`. Rode a partir da
**raiz do repositório**:

```bash
# a partir da raiz do repositório (SIMPLE/), com o venv ativado (ou use .venv/bin/python3)
PYTHONPATH=src .venv/bin/python3 src/simple/cli/teleop_decoupled_wbc.py \
  <env_id> \
  --no-headless
```

Exemplo real, com a task de prateleira-para-mesa:

```bash
PYTHONPATH=src .venv/bin/python3 src/simple/cli/teleop_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --no-headless
```

Flags mais usadas (todas opcionais, com valor padrão razoável):

| Flag | Default | Pra que serve |
| :--- | :--- | :--- |
| `--target` | nenhum | Objeto-alvo, só faz sentido em tasks com DR `target` (ex. `graspnet1b:0`). Tasks como a de totes não usam isso — é ignorado silenciosamente se passado. |
| `--no-headless` | (headless=True) | Abre a janela local do MuJoCo (útil pra acompanhar do PC além do headset). |
| `--record` | desligado | Ativa a gravação dos episódios (seção 10). |
| `--num-episodes` | 100 | Quantos episódios gravar antes de encerrar sozinho (só importa com `--record`). |
| `--dr-level` | 0 | Nível de randomização de domínio. Comece em 0. |
| `--save-dir` | `data/teleop_decoupled_wbc` | Onde os dados gravados são salvos (relativo à raiz do repo, a menos que você passe um caminho absoluto). |

Ao subir com sucesso, o terminal mostra algo como:

```
[VuerDecoupled] TeleVuer started. Open https://<IP_DO_PC>:8012 in the Meta Quest browser.
```

Guarde esse endereço — é o que você vai digitar no navegador do headset.

> Nota sobre o `--target`: cada task tem seus próprios requisitos de DR — se você não sabe se
> uma task específica precisa dele, olhe o arquivo dela em `src/simple/tasks/` (o bloco
> `dr_cfgs`) antes de rodar.

---

## 9. Conectar o headset Meta Quest 3

1. Coloque o headset e abra o **navegador do Meta Quest** (Meta Quest Browser).
2. Digite o endereço mostrado no terminal (ex.: `https://192.168.1.10:8012`).
3. Vai aparecer um aviso de segurança (o certificado é autoassinado, gerado localmente pelo
   TeleVuer) — clique em **Avançado** → **Prosseguir de forma insegura** (ou texto
   equivalente).
4. Clique em **Entrar em VR** / **Immersive Mode** no navegador. A partir daqui você está
   vendo pela câmera estéreo do robô.

O robô começa suspenso no ar e faz um "pouso" (elastic band) sozinho até tocar o chão.

---

## 10. Controlando o robô

- **Botão B (controle esquerdo)**: ativa/desativa o controle dos braços pelo teleop (toggle).
  Você vai ouvir um som e ver `Teleop activated` nos logs quando ligar.
- **Mãos**: mova as mãos normalmente — braços e pulsos do robô seguem de forma simétrica.
- **Thumbstick esquerdo**: locomoção da base (andar pela cena).
- **Trigger (gatilho)**: fecha o dedo indicador.
- **Squeeze (aperto lateral)**: fecha os demais dedos.
- **Squeeze esquerdo + direito ao mesmo tempo**: gesto de **reset** — reinicia o episódio
  (a cena é re-randomizada).

Pra desligar o controle dos braços, aperte **B** esquerdo de novo.

---

## 11. Gravando dados

Adicione `--record` (e opcionalmente `--num-episodes`) ao comando da seção 8:

```bash
# a partir da raiz do repositório (SIMPLE/)
PYTHONPATH=src .venv/bin/python3 src/simple/cli/teleop_decoupled_wbc.py \
  simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 \
  --no-headless --record --num-episodes 10
```

Assim que o ambiente carrega, o terminal pergunta qual instrução (prompt em linguagem natural)
deve ser gravada junto com essa sessão, mostrando o padrão da task entre colchetes:

```
[Record] Instrução da tarefa [pegue o tote azul da estante e coloque na mesa]:
```

Aperte **Enter** pra manter o padrão, ou digite um texto diferente e aperte Enter pra usar sua
própria instrução nessa sessão (ex.: pra descrever uma variação específica do que você vai
gravar). O texto escolhido fica salvo em `metadata.json` (`task_prompt`) e é o que vai junto de
cada frame do dataset.

O que acontece durante a gravação:

- O robô só começa a gravar de fato depois de pousar **e** você apertar o botão B pra ativar
  o teleop (estado interno `WAITING_FOR_LANDING` → `RECORDING`).
- Enquanto grava, um HUD ao vivo é exibido (pra tasks que oferecem essa info, como a de
  totes) mostrando contagens tipo "X totes na estante / Y totes na mesa".
- Quando a task é concluída com sucesso (critério específico de cada task — ver o arquivo
  dela em `src/simple/tasks/`), o episódio é salvo automaticamente e a cena reseta sozinha
  pro próximo episódio.
- Se algo der errado no meio (ex.: um objeto cair no chão, ou você apertar o gesto de reset),
  o episódio em andamento é **descartado** (não salvo) e a cena reseta — isso é intencional,
  pra não poluir o dataset com episódios corrompidos.
- `Ctrl+C` interrompe a sessão; se um episódio estava em gravação, ele tenta salvar antes de
  sair.

### Onde os dados ficam salvos

```
data/teleop_decoupled_wbc/<env_id>/level-<dr_level>/sessions/<timestamp>__<operador>/
```

(caminho relativo à raiz do repositório, a menos que você tenha passado `--save-dir` com
outro destino). **Cada execução com `--record` cria uma pasta de sessão nova** (timestamp no
formato `AAAAMMDD_HHMMSS`), autocontida — não acumula episódios de sessões diferentes na mesma
pasta. Dentro dela:

- Os dados seguem o **formato LeRobot/GR00T** (vídeos + parquet + metadados) — já é o formato
  final, não existe um passo de conversão separado depois.
- `meta/episodes.jsonl` tem uma entrada por episódio, incluindo um campo extra
  `environment_config` com o estado completo da task naquele episódio (posições dos objetos,
  randomização aplicada, etc.) — útil pra depois auditar ou filtrar episódios.
- `metadata.json` (na raiz da sessão, não dentro de `meta/`) guarda o operador, o timestamp, o
  `env_id`, o nível de DR e o **status da sessão no pipeline**: `raw_captured` logo após a
  gravação, depois `uploaded` assim que você rodar o upload da seção 12. Esse arquivo é o que a
  validação do `upload-teleop-session` confere antes de subir — não edite ele manualmente.

### Guias específicos por task

Cada task pode ter particularidades de captura (pontos de parada recomendados, riscos
conhecidos, etc.). Antes de gravar uma sessão "de verdade" numa task nova, procure se existe
um guia dedicado em `docs/teleop_simple_study/` — por exemplo,
`teleop_shelf_to_table_vr_guide.md` pra task de prateleira-para-mesa. Se não existir um pra
sua task, siga o fluxo genérico deste documento e valide com calma (visual, alcance dos
braços, timing de sucesso) antes de gravar muitos episódios.

---

## 12. Enviando os dados pro Hugging Face

Existe um comando dedicado pra isso, `upload-teleop-session` — ele valida a sessão (confere se
tem pelo menos 1 episódio, se os arquivos de vídeo/parquet batem com o esperado, se nada ficou
com 0 bytes) antes de subir, e evita que você suba uma sessão vazia ou corrompida por engano.
Ele lê `HF_TOKEN` e `HF_REPO_RAW` direto do seu `.env` (seção 4) — não precisa de `hf auth
login` nem de passar `--repo-id` na mão.

**Os repositórios (`HF_REPO_RAW`/`HF_REPO_RENDERED`) já devem existir** — criação de
repositório é uma ação administrativa feita uma vez só (ver
`docs/source/tutorials/teleop_hf_pipeline.md`), não algo que você faz por sessão. Se o comando
abaixo falhar dizendo que o repo não existe, avise quem administra o dataset antes de tentar
criar um por conta própria.

### 12.1 Conferir o caminho antes de subir (opcional, mas recomendado na primeira vez)

```bash
# a partir da raiz do repositório (SIMPLE/), com o venv ativado
upload-teleop-session data/teleop_decoupled_wbc/<env_id>/level-<dr_level>/sessions/<timestamp>__<operador> --dry-run
```

Isso só valida a sessão e imprime pra onde ela iria (`raw/<operador>/<timestamp>__<env>__level-<n>/`)
sem subir nada.

### 12.2 Subir de verdade

```bash
# a partir da raiz do repositório (SIMPLE/), com o venv ativado
upload-teleop-session data/teleop_decoupled_wbc/<env_id>/level-<dr_level>/sessions/<timestamp>__<operador>
```

Ao terminar, o `metadata.json` local da sessão é atualizado pra `status: "uploaded"` — é assim
que o `sync-and-render` (rodado depois, numa workstation com Isaac Sim) sabe quais sessões já
estão prontas pra serem re-renderizadas.

**Erros comuns:**

| Erro | Causa | O que fazer |
| :--- | :--- | :--- |
| `Session validation failed: ... total_episodes=0` | Sessão gravada sem nenhum episódio salvo (ex.: `Ctrl+C` antes do primeiro sucesso) | Não precisa subir — apague a pasta da sessão ou ignore |
| `Session validation failed: ... num_episodes=None/0` no `metadata.json` | Mesma causa acima, ou a gravação foi interrompida de forma anormal (crash, não `Ctrl+C`) | Confira quantos vídeos existem em `videos/`; se for 0, descarte a sessão |
| `No repo id given` | `HF_REPO_RAW` não está no seu `.env` | Adicione `HF_REPO_RAW=...` ao `.env` (seção 4) |
| `No Hugging Face token found` | `HF_TOKEN` vazio ou ausente no `.env` | Gere um token com permissão de escrita e adicione ao `.env` |
| Erro 401/403 do Hugging Face | Token sem permissão de escrita nesse repo específico | Gere um novo token com o escopo certo — não compartilhe seu token com outras pessoas |

---

## 13. Problemas comuns

- **`TeleVuer port 8012 is already in use`**: uma sessão anterior não fechou direito. Rode
  `lsof -i :8012` (de qualquer diretório) pra achar o processo e finalize-o (`kill <PID>`).
- **Tela preta no headset / imagem não chega**: normalmente é o buffer estéreo com resolução
  incompatível — já foi corrigido no `vuer_decoupled_agent.py`, mas se voltar a acontecer,
  veja a seção 2.2 do `MetaQuest_Teleoperation_Guide.md` (raiz do repositório).
- **`GLFWError: GLFW library is not initialized`**: rodando headless num servidor sem
  display. Garanta `export MUJOCO_GL=egl` (já vem no `.env.sample`).
- **Erros de instalação (`uv sync`, CuRobo, Isaac Sim, etc.)**: consulte
  `docs/source/troubleshooting.md` (a partir da raiz do repositório) — tem uma lista grande
  de erros já resolvidos antes, com a causa e o comando de correção.
- **Headset não conecta ao IP mostrado no terminal**: confirme rede Wi-Fi igual (seção 7);
  redes corporativas com "isolamento de cliente" (client isolation) costumam bloquear esse
  tipo de conexão P2P — nesse caso, use um roteador/hotspot dedicado.

---

## 14. Pra onde ir depois

- Estrutura geral de pastas do repo: `docs/teleop_simple_study/folder_structure.md`.
- Como funciona o fluxo de trabalho ponta a ponta (não só teleop): `docs/teleop_simple_study/workflow.md`.
- Como adicionar novos objetos/cenas: `docs/teleop_simple_study/adding_objects_guide.md` e
  `docs/teleop_simple_study/usd_asset_to_simple_teleop.md`.
- Detalhes técnicos do pipeline raw → upload → renderização (`sync-and-render`,
  `render-decoupled-wbc`), schema do `metadata.json`, e como o estágio de renderização
  fotorrealista funciona numa workstation com Isaac Sim:
  `docs/source/tutorials/teleop_hf_pipeline.md`.
- Dúvidas sobre uma task específica: leia o arquivo dela em `src/simple/tasks/` primeiro —
  os comentários no código costumam documentar decisões e placeholders conhecidos.
- Qualquer coisa que não esteja clara ou que pareça errada neste guia, me avise — prefiro
  corrigir o documento do que você perder tempo travado em algo que eu expliquei mal.
