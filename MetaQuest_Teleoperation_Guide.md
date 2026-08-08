# Guia de Integração e Uso: Teleoperação com Meta Quest 3 (Vuer) no SIMPLE

Este documento detalha o processo de integração, as modificações realizadas no código-fonte e o guia passo a passo de como instalar, configurar e executar a teleoperação do robô G1 no simulador SIMPLE utilizando o **Meta Quest 3** através da biblioteca **TeleVuer** (framework WebXR Vuer).

---

## 1. Visão Geral da Arquitetura

Originalmente, a teleoperação no SIMPLE suportava apenas o headset PICO via `XRoboToolkit` e uma conexão TCP direta para streaming (H.264).
O objetivo foi substituir esse acoplamento fechado por uma solução agnóstica de hardware (WebXR) usando o **TeleVuer**, mantendo intacto o controlador de corpo inteiro (`decoupled_wbc`).

### Novos Componentes Desenvolvidos:
1. **`VuerStreamer`**: Estende a interface `BaseStreamer`. Responsável por ler os dados brutos da memória compartilhada do TeleVuer (pose da cabeça e mãos no padrão OpenXR), converter para o referencial esperado pelo `WristsPreProcessor` (z-up relativo ao headset) e mapear os botões do Meta Quest.
2. **`VuerDecoupledAgent`**: Substitui o `PicoDecoupledAgent`. Gerencia o ciclo de vida do servidor Vuer, injeta o `VuerStreamer` no `TeleopPolicy` e processa o streaming visual renderizado pelo simulador, adaptando-o para a interface WebXR.

---

## 2. Principais Modificações no Código

### 2.1. Inversão do Braço Direito (`wrists.py`)
No código original, o `WristsPreProcessor` possuía uma regra estrita: qualquer dispositivo que não fosse `"pico"` teria o eixo Z do pulso direito invertido, quebrando a cinemática da teleoperação.
**Solução**: Inclusão explícita do device `"vuer"`.
*Arquivo*: `third_party/decoupled_wbc/control/teleop/pre_processor/wrists/wrists.py`

```python
# Correção aplicada:
if self.control_device == "pico" or self.control_device == "vuer":
    relative_pose = get_relative_pose(arm_pose, head_pose)
else:
    relative_pose = get_relative_pose(arm_pose, head_pose)
    if "right" in self.side:
        relative_pose = apply_rotation(relative_pose, [0, 0, 1], np.pi)
```

### 2.2. Resolução da "Tela Preta" no Vuer (`vuer_decoupled_agent.py`)
O simulador renderizava a câmera estéreo em `360x1280` (`head_stereo`), mas o buffer de memória compartilhada do TeleVuer (`img2display_shm`) esperava estritamente `480x1280`. Isso gerava um `ValueError` silencioso numa thread de background do TeleVuer, impedindo as imagens de chegarem ao headset.
**Solução**: Redimensionamento dinâmico (`cv2.resize`) utilizando a propriedade `img_shape` da instância base do TeleVuer.

```python
# Em VuerDecoupledAgent._push_stereo_frame()
stereo_bgr = np.concatenate([left_bgr, right_bgr], axis=1)

# Redimensionamento dinâmico para evitar crash no buffer do TeleVuer
target_h, target_w, _ = self._tv_wrapper.tvuer.img_shape
if stereo_bgr.shape[:2] != (target_h, target_w):
    stereo_bgr = cv2.resize(stereo_bgr, (target_w, target_h))

self._tv_wrapper.render_to_xr(stereo_bgr)
```

### 2.3. Resolução de Dependências (Docker e `pyproject.toml`)
- Adição de `televuer = {path = "third_party/televuer", editable = true}` no `pyproject.toml`.
- O ambiente Docker isolado falhava ao instalar a dependência `xrobotoolkit-sdk` por ausência da biblioteca de build `setuptools`. Injetamos o `setuptools` de forma pre-buildada no ambiente virtual e usamos `--no-build-isolation-package`.
- Configuração correta do pacote `qpsolvers` usando a flag extra `[open_source_solvers]` para garantir a existência de backends para cálculo IK.

---

## 3. Instalação e Configuração do Ambiente

### 3.1. Clonagem Inicial (Com Submódulos)
Como o projeto agora depende do repositório TeleVuer acoplado em `third_party`, todos que forem clonar precisam buscar os submódulos:

```bash
git clone --recursive <url_do_seu_repositorio>
# Ou se já tiver clonado:
git submodule update --init --recursive
```

### 3.2 Instalacao do ambiente

Intale o ambiente virtual seguindo as orientacoes do arquivo README.md. Se ja instalado, va para o diretorio do televuer e rode: `uv pip install -e .`

---

## 4. Como Executar a Teleoperação (Passo a Passo)

### 4.1. Iniciar o Script do Agente
Com o container Docker e o ambiente `.venv` ativados, inicie a simulação pelo script de CLI, especificando o target da simulação com o MuJoCo:

```bash
python src/simple/cli/teleop_decoupled_wbc.py simple/G1WholebodyXMoveBendPickTeleop-v0 \
    --target=graspnet1b:0 \
    --sim-mode=mujoco \
    --record \
    --no-headless
```
*Nota: a primeira inicialização baixará o pipeline do modelo ONNX. Fique atento aos logs.*

Ao rodar com sucesso, o terminal exibirá:
`[VuerDecoupled] TeleVuer started. Open https://<PC_IP>:8012 in the Meta Quest browser.`

### 4.2. Conexão pelo Headset (Meta Quest 3)
1. Certifique-se de que o PC Host e o Meta Quest 3 estão na **mesma rede Wi-Fi** (preferencialmente 5GHz ou via roteador dedicado).
2. Abra o Meta Quest Browser no headset.
3. Digite o endereço IP mostrado no terminal (ex: `https://192.168.1.10:8012`).
4. Caso surja um aviso de segurança (Devido ao certificado SSL autoassinado), clique em **Avançado** e **Prosseguir de forma insegura**.
5. Clique no botão de **Entrar em VR (Enter VR)** ou **Immersive Mode** no navegador. Você agora estará vendo pelo robô.

### 4.3. Controlando o Robô (Mapeamento de Botões)
O robô iniciará no ar e fará o *landing* inicial no chão com uma elástico virtual.
Para engajar os braços do robô com os controladores do VR, você precisa dar o comando de **ativação**:

- **Botão de Ativação (Toggle)**: Pressione o botão **B** (no controle esquerdo).
  - Você ouvirá um sinal e verá nos logs: `Teleop activated`.
- **Movimento**: Movimente suas mãos. Os braços e pulsos do robô deverão seguir os seus de forma simétrica.
- **Navegação da Base**: Use o **Thumbstick Esquerdo**.
- **Manipulação dos Dedos**: 
  - Gatilho (`Trigger`): Fecha indicador.
  - Aperto lateral (`Squeeze`): Fecha demais dedos.

Para interromper ou desativar o engajamento dos braços, basta pressionar o botão **B** esquerdo novamente.
