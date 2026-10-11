# Debug de conexão VR (televuer)

Notas de uma sessão de debug em que a teleoperação via headset (Meta Quest,
pipeline `teleop-decoupled-wbc` + `VuerDecoupledAgent`) não conseguia
estabelecer conexão real com o VR: a página carregava, a sessão WebXR
entrava (grid padrão do Three.js visível no headset), mas nenhuma pose,
botão ou imagem chegava ao processo Python.

## Sintoma

* `scripts/test_vuer_streamer.py` conectava sem erro e passava nas
  verificações estáticas de shape/SE3 (que usam poses de fallback, então
  sempre passam mesmo sem headset real).
* No loop ao vivo, `HEAD` ficava travado em zero e `LEFT`/`RIGHT` travados
  exatamente nos valores constantes de fallback (`CONST_LEFT_ARM_POSE` /
  `CONST_RIGHT_ARM_POSE`) — nunca uma leitura real do controle, mesmo após
  clicar **"Enter VR"** no navegador do headset.
* Nenhuma imagem em primeira pessoa aparecia no headset.

## Causa raiz

O servidor `vuer` (usado pelo `televuer`) imprime no terminal, no momento
em que um cliente WebSocket conecta de verdade:

```
websocket is connected. id:{ws_id}
Uplink task running. id:{ws_id}
default socket worker is up, adding clientEvents
```

(`vuer/server.py`, respectivamente linhas ~609, ~570, ~416 na versão
`0.0.60`). **Essas linhas nunca apareciam** — ou seja, o WebSocket de dados
nunca completava o handshake, independentemente de a sessão WebXR ter
entrado visualmente (isso é client-side puro, não depende do socket).

Inspecionando o navegador do Quest via `chrome://inspect#devices` (USB
debugging, veja seção abaixo), o console mostrava:

```
create-or-join.js:48 WebSocket connection to 'wss://<PC_IP>/' failed
+Page.tsx:122 Max reconnect attempts of 3 exceeded
```

Note a ausência da porta: o front-end do Vuer estava tentando abrir o
WebSocket em `wss://<PC_IP>/` (porta padrão `443`), em vez de herdar a
porta `8012` em que o servidor realmente escuta. Como nada escuta em 443,
a conexão falha, tenta de novo 3x e desiste — silenciosamente, sem nenhum
erro visível na UI do headset.

## Fix

Abrir a URL no navegador do headset **com a porta do WebSocket explícita**
via query param `ws=`:

```
https://<PC_IP>:8012/?ws=wss://<PC_IP>:8012
```

O parâmetro `?ws=wss://<ip>:<porta>` diz ao cliente Vuer exatamente onde
abrir o WebSocket, em vez de depender da inferência automática (que aqui
se mostrou quebrada mesmo em uma rede com uma única interface — não é
apenas um problema de hosts multi-NIC).

> Isso também está documentado, para o cenário específico de hosts com
> múltiplas interfaces de rede, em commits da branch `industrial_env`
> (`aa00a82`). Nesta investigação o mesmo sintoma apareceu numa rede de
> interface única, então trate `?ws=wss://<ip>:<porta>` como a forma
> padrão de abrir a conexão, não como um caso especial.

## Problema relacionado (já corrigido): crash de pose antes da 1ª conexão

Antes do primeiro evento `CAMERA_MOVE` do headset, as matrizes de pose
(`head_pose`, `left_arm_pose`, `right_arm_pose`) do `televuer` vêm
zero-inicializadas, o que causa `ValueError: Non-positive determinant` em
qualquer código que tente decompor essas matrizes (ex.: `scipy.spatial
.transform.Rotation.from_matrix`).

* No submódulo `third_party/televuer`, `get_headset_relative_wrist_poses()`
  já tem fallback para poses constantes quando o determinante é inválido
  (fix trazido do `industrial_env`, submódulo atualizado para o commit
  `b379965` — `feat: add fallback to default constants if matrices are
  not valid/uninitialized`).
* Em `src/simple/teleop/vuer/vuer_streamer.py`, `_safe_get_wrist_poses()`
  já implementa uma proteção equivalente (e mais rigorosa: valida SE3
  completo e cai para a **última pose válida conhecida**, não uma
  constante fixa) — este guard já existia independentemente nesta base de
  código, então nenhuma mudança adicional foi necessária ali.
* `scripts/test_vuer_streamer.py` (`_mat_str`) não tinha essa proteção só
  para fins de exibição/diagnóstico — corrigido para mostrar
  `"not yet valid — det=..."` em vez de crashar quando a pose ainda não é
  válida.

Esse guard evita o crash, mas **não resolve** o problema de conexão em si
— ele só permite que o diagnóstico continue rodando para você chegar até
a causa real (a porta do WebSocket).

## Como depurar esse tipo de problema (canal de controle vs. canal de imagem)

1. **Isolar o canal de controle (pose/WebXR) do canal de imagem.** Rodar
   `scripts/test_vuer_streamer.py` — ele sobe só `TeleVuerWrapper` +
   `VuerStreamer`, com `display_mode="pass-through"`, `zmq=False`,
   `webrtc=False`, então não depende de imagem nenhuma. Se as poses
   mudarem ao mover os controles, o canal de dados está OK.

2. **Ver se o WebSocket de dados realmente conecta.** No terminal onde o
   script Python está rodando, procure pelas linhas de log do `vuer`
   (`websocket is connected. id:...`). Se nunca aparecerem, o problema é
   de conexão, não de imagem.

3. **Inspecionar o navegador do headset via `chrome://inspect` (USB):**
   * Habilitar Developer Mode no Quest (app Meta Quest no celular →
     Dispositivos → seu headset → Developer Mode).
   * Conectar via cabo USB e aceitar o prompt de depuração USB dentro do
     headset.
   * No PC, `adb devices -l` deve mostrar o headset como `device` (não
     `unauthorized`) — se aparecer `unauthorized`, aceite o prompt dentro
     do headset e rode de novo.
   * Abrir `chromium-browser chrome://inspect#devices` (precisa de um
     navegador baseado em Chromium no PC — Firefox não funciona aqui,
     mesmo sendo só para inspecionar, porque o protocolo de depuração
     remota é específico do Chromium, que é a base do navegador do Quest).
   * Com uma aba aberta no navegador do Quest, ela aparece listada em
     "Remote Target" — clique em **inspect**.
   * Aba **Console**: procure erros relacionados a `WebSocket`, `wss://`,
     `SSL`/`certificate`.
   * Aba **Network**, filtro **WS**: veja se a conexão finaliza como
     handshake bem-sucedido (`101 Switching Protocols`) ou falha
     (`Finished`/`(failed)` com `0.0 kB` transferidos).

4. **Só depois de confirmar o canal de controle**, testar o canal de
   imagem: o pipeline real (`teleop_decoupled_wbc.py` →
   `VuerDecoupledAgent`) usa `zmq=True, webrtc=False`
   (`src/simple/agents/vuer_decoupled_agent.py:127-134`), diferente das
   flags `--webrtc` expostas em `src/simple/cli/teleop.py`/`datagen.py`,
   que pertencem a outro caminho (env-based) não usado por
   `teleop_decoupled_wbc`. O frame estéreo é montado e enviado em
   `vuer_decoupled_agent.py:298-306`
   (`self._tv_wrapper.render_to_xr(stereo_bgr)`) — um bom ponto para
   instrumentar caso a imagem ainda não apareça depois do canal de
   controle estar funcionando.
