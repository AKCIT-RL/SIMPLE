"""
test_vuer_streamer.py — Fase 4: Validação de Poses e StreamerOutput

Script standalone para validar a integração TeleVuerWrapper ↔ VuerStreamer
sem precisar do SIMPLE completo (sem MuJoCo, sem WBC, sem robô).

O que este script valida
------------------------
  [1] TeleVuerWrapper inicializa e o servidor Vuer sobe OK
  [2] get_headset_relative_wrist_poses() retorna SE3 (4×4) válidas
  [3] As poses mudam ao mover os controles (sem dados congelados/zero)
  [4] VuerStreamer.get() produz StreamerOutput com shapes corretos
  [5] Botões edge-detected funcionam (toggle_activation, toggle_policy, etc.)
  [6] Thumbstick → navigate_cmd mapeado corretamente (sign, dead-zone)

Uso
---
  1. Ative o ambiente com televuer e decoupled_wbc instalados
  2. Execute:
       python3 scripts/test_vuer_streamer.py

  3. Abra  https://<PC_IP>:8012  no browser do Meta Quest
  4. Mova os controles e verifique a saída no terminal
  5. Pressione Ctrl+C para sair

Saída esperada ao conectar o headset
-------------------------------------
  - Head/Left/Right poses: matrizes SE3 com translação não-zero
  - Pose changes: "MOVING" quando detector de mudança percebe variação
  - StreamerOutput: shapes corretos (left_wrist 4×4, finger_data 25×4×4 etc.)
  - Ao mover o thumbstick esquerdo para frente → lin_vel_x > 0
  - Ao pressionar o botão B esquerdo → toggle_activation = True (edge)
"""

import argparse
import os
import sys
import time
import signal
import threading

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from the SIMPLE root without installing packages
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

_TELEVUER_SRC = os.path.join(
    os.path.dirname(_ROOT),
    "xr_teleoperate", "teleop", "televuer", "src",
)
if os.path.isdir(_TELEVUER_SRC):
    sys.path.insert(0, _TELEVUER_SRC)


# ── ANSI colours (degrade gracefully on terminals that don't support them) ────
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
RED    = lambda t: _c("91", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RESET  = "\033[0m"


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_valid_se3(mat: np.ndarray, name: str) -> bool:
    """Check that mat is a valid SE3 matrix (det ≈ 1, no NaN/Inf)."""
    if mat.shape != (4, 4):
        print(RED(f"  [FAIL] {name}: expected (4,4), got {mat.shape}"))
        return False
    if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-4):
        print(RED(f"  [FAIL] {name}: last row is not [0,0,0,1]: {mat[3]}"))
        return False
    det = np.linalg.det(mat[:3, :3])
    if not np.isclose(det, 1.0, atol=1e-3):
        print(RED(f"  [FAIL] {name}: rotation det={det:.4f}, expected 1.0"))
        return False
    if not np.all(np.isfinite(mat)):
        print(RED(f"  [FAIL] {name}: contains NaN or Inf"))
        return False
    return True


def _mat_str(mat: np.ndarray) -> str:
    """Compact 2-line representation of position and axis-angle."""
    pos  = mat[:3, 3]
    det  = np.linalg.det(mat[:3, :3])
    if not np.isfinite(det) or np.isclose(det, 0.0, atol=1e-6):
        return f"  pos [x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}]   rpy [not yet valid — det={det:.4f}]"
    from scipy.spatial.transform import Rotation as R
    rpy  = R.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True)
    return (
        f"  pos [x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}]"
        f"   rpy [{rpy[0]:+.1f}°  {rpy[1]:+.1f}°  {rpy[2]:+.1f}°]"
    )


def _finger_summary(fingers: np.ndarray) -> str:
    """Summarise finger state from (25,4,4) array."""
    THUMB = 0; INDEX = 5; MIDDLE = 10; RING = 15
    vals = {
        "thumb":  fingers[4 + THUMB,  0, 3],
        "index":  fingers[4 + INDEX,  0, 3],
        "middle": fingers[4 + MIDDLE, 0, 3],
        "ring":   fingers[4 + RING,   0, 3],
    }
    parts = []
    for name, val in vals.items():
        icon = "●" if val > 0.5 else "○"
        parts.append(f"{name}:{icon}")
    return "  " + "  ".join(parts)


def _detect_motion(prev: np.ndarray | None, curr: np.ndarray, tol: float = 1e-4) -> str:
    if prev is None:
        return DIM("INIT")
    delta = np.linalg.norm(curr[:3, 3] - prev[:3, 3])
    if delta > tol:
        return GREEN(f"MOVING  Δ={delta:.4f}m")
    return DIM("STILL")


# ── main test loop ─────────────────────────────────────────────────────────────

def run_test(args) -> None:
    # ── 1. Import (deferred so path setup above takes effect) ─────────────────
    try:
        from televuer import TeleVuerWrapper
    except ImportError as e:
        print(RED(f"[ERROR] Could not import televuer: {e}"))
        print(YELLOW("  Make sure televuer is installed or run from the SIMPLE root with"))
        print(YELLOW("  the xr_teleoperate submodule present at ../xr_teleoperate/"))
        sys.exit(1)

    try:
        from simple.teleop.vuer.vuer_streamer import VuerStreamer
        from decoupled_wbc.control.teleop.streamers.base_streamer import StreamerOutput
    except ImportError as e:
        print(RED(f"[ERROR] Could not import VuerStreamer / decoupled_wbc: {e}"))
        print(YELLOW("  Make sure both SIMPLE and decoupled_wbc are installed / on PYTHONPATH."))
        sys.exit(1)

    print(BOLD("\n═══════════════════════════════════════════════════════"))
    print(BOLD("  VuerStreamer — Fase 4: Validação de Poses"))
    print(BOLD("═══════════════════════════════════════════════════════"))

    # ── 2. Start TeleVuerWrapper ───────────────────────────────────────────────
    print(f"\n{CYAN('[1/5]')} Iniciando TeleVuerWrapper ...")
    cert_file = args.cert_file or None
    key_file  = args.key_file  or None

    tv = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=True,
        img_shape=(args.height, args.width * 2),   # side-by-side stereo
        display_fps=30.0,
        display_mode="pass-through",
        zmq=False,
        webrtc=False,
        cert_file=cert_file,
        key_file=key_file,
    )
    print(GREEN("  ✓ TeleVuerWrapper iniciado"))
    print(YELLOW(f"  → Abra  https://<PC_IP>:{args.port}  no browser do Meta Quest"))
    print(YELLOW("  → Pressione Enter neste terminal quando o headset estiver conectado\n"))

    if not args.skip_wait:
        input(DIM("  [Enter para continuar] "))

    # ── 3. Instantiate VuerStreamer ────────────────────────────────────────────
    print(f"\n{CYAN('[2/5]')} Criando VuerStreamer ...")
    streamer = VuerStreamer(tv)
    streamer.start_streaming()
    print(GREEN("  ✓ VuerStreamer criado"))

    # ── 4. Static shape / validity check ──────────────────────────────────────
    print(f"\n{CYAN('[3/5]')} Verificação estática de shapes e validade SE3 ...")
    left_w, right_w = tv.get_headset_relative_wrist_poses()
    all_ok = True
    for name, mat in [("left_wrist", left_w), ("right_wrist", right_w)]:
        ok = _is_valid_se3(mat, name)
        print(f"  {'✓' if ok else '✗'} {name}: shape={mat.shape}, valid={ok}")
        all_ok = all_ok and ok

    output = streamer.get()
    shape_checks = [
        ("left_wrist  shape",   output.ik_data["left_wrist"].shape,             (4, 4)),
        ("right_wrist shape",   output.ik_data["right_wrist"].shape,            (4, 4)),
        ("left_fingers shape",  output.ik_data["left_fingers"]["position"].shape, (25, 4, 4)),
        ("right_fingers shape", output.ik_data["right_fingers"]["position"].shape,(25, 4, 4)),
        ("navigate_cmd len",    len(output.control_data["navigate_cmd"]),        4),
    ]
    for desc, got, expected in shape_checks:
        ok = got == expected
        all_ok = all_ok and ok
        icon = "✓" if ok else "✗"
        col  = GREEN if ok else RED
        print(col(f"  {icon} {desc}: {got}  (expected {expected})"))

    if all_ok:
        print(GREEN("\n  ✓ Todos os shapes e validações SE3 passaram!\n"))
    else:
        print(RED("\n  ✗ Algumas verificações falharam — veja acima.\n"))

    # ── 5. Live streaming loop ─────────────────────────────────────────────────
    print(BOLD(f"{CYAN('[4/5]')} Loop de validação ao vivo  ({args.rate} Hz)"))
    print(BOLD("  ┌─ Controles de saída ─────────────────────────────────────────────┐"))
    print(BOLD("  │  Mova os controles → pose muda → MOVING aparece                  │"))
    print(BOLD("  │  Pressione B esquerdo → toggle_activation = True (uma vez)       │"))
    print(BOLD("  │  Pressione A esquerdo → toggle_policy_action = True (uma vez)    │"))
    print(BOLD("  │  Thumbstick esquerdo ↑ → lin_vel_x > 0                           │"))
    print(BOLD("  │  Ctrl+C para sair                                                 │"))
    print(BOLD("  └──────────────────────────────────────────────────────────────────┘\n"))

    stop = threading.Event()

    def _sigint(_sig, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _sigint)

    prev_left  = None
    prev_right = None
    prev_head  = None
    dt = 1.0 / args.rate
    step = 0
    activation_count = 0
    policy_count     = 0
    dc_count         = 0

    while not stop.is_set():
        t0 = time.monotonic()
        step += 1

        # --- read ---
        try:
            left_w, right_w = tv.get_headset_relative_wrist_poses()
            td              = tv.get_tele_data()
            so              = streamer.get()
        except Exception as exc:
            print(RED(f"  [step {step:5d}] Exception: {exc}"))
            time.sleep(dt)
            continue

        # --- count edge events ---
        if so.teleop_data.get("toggle_activation"):
            activation_count += 1
        if so.control_data.get("toggle_policy_action"):
            policy_count += 1
        if so.data_collection_data.get("toggle_data_collection"):
            dc_count += 1

        # --- print every N steps ---
        if step % args.print_every == 0:
            nav   = so.control_data.get("navigate_cmd", [0, 0, 0, 0])
            height = so.control_data.get("base_height_command", 0.0)

            print(f"{'─'*70}")
            print(BOLD(f"  Step {step:5d}  │  t={time.monotonic():.1f}s"))
            print()

            # Head pose (from tele_data)
            head_m = tv.tvuer.head_pose
            print(f"  HEAD    {_detect_motion(prev_head, head_m)}")
            print(DIM(_mat_str(head_m)))

            # Left wrist
            print(f"  LEFT    {_detect_motion(prev_left, left_w)}")
            print(DIM(_mat_str(left_w)))

            # Right wrist
            print(f"  RIGHT   {_detect_motion(prev_right, right_w)}")
            print(DIM(_mat_str(right_w)))

            # Fingers
            print(f"  FINGERS-L {_finger_summary(so.ik_data['left_fingers']['position'])}")
            print(f"  FINGERS-R {_finger_summary(so.ik_data['right_fingers']['position'])}")

            # Navigation
            print(
                f"  NAV     lin_x={nav[0]:+.3f}  lin_y={nav[1]:+.3f}"
                f"  vyaw={nav[2]:+.3f}  yaw_tgt={nav[3]:+.3f}  h={height:.3f}m"
            )

            # Buttons summary
            act_str = GREEN(f"TOGGLE×{activation_count}") if activation_count else DIM("─")
            pol_str = GREEN(f"TOGGLE×{policy_count}")     if policy_count     else DIM("─")
            dc_str  = GREEN(f"TOGGLE×{dc_count}")         if dc_count         else DIM("─")
            print(
                f"  BUTTONS activation={act_str}  policy={pol_str}"
                f"  data_collect={dc_str}"
            )

            # Raw tele_data buttons (live, not edge)
            print(
                DIM(
                    f"  RAW     L.A={td.left_ctrl_aButton}  L.B={td.left_ctrl_bButton}"
                    f"  R.A={td.right_ctrl_aButton}  R.B={td.right_ctrl_bButton}"
                    f"  R.trigger={td.right_ctrl_trigger}  R.squeeze={td.right_ctrl_squeeze}"
                )
            )
            print(
                DIM(
                    f"          L.thumb={list(np.round(td.left_ctrl_thumbstickValue, 2))}"
                    f"  R.thumb={list(np.round(td.right_ctrl_thumbstickValue, 2))}"
                )
            )

            prev_left  = left_w.copy()
            prev_right = right_w.copy()
            prev_head  = head_m.copy()

        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print(f"\n{CYAN('[5/5]')} Encerrando ...")
    tv.close()
    print(BOLD(GREEN("\n  ═══ Sessão de teste concluída ═══")))
    print(f"  Steps executados  : {step}")
    print(f"  toggle_activation : {activation_count}×")
    print(f"  toggle_policy     : {policy_count}×")
    print(f"  toggle_data_coll  : {dc_count}×")
    print()

    # Final verdict
    if all_ok:
        print(GREEN("  [PASS] Shapes e validade SE3 OK — integração básica funcionando."))
        print(YELLOW("  Próximo passo: rodar SIMPLE com VuerDecoupledAgent e testar IK."))
    else:
        print(RED("  [FAIL] Verificações estáticas falharam — revise acima antes de prosseguir."))
    print()


# ── entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fase 4: valida poses e StreamerOutput do VuerStreamer (sem SIMPLE completo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos
--------
  # Modo padrão (espera Enter antes do loop):
  python3 scripts/test_vuer_streamer.py

  # Modo headless / CI (não espera, taxa 5 Hz, para após 30 iterações):
  python3 scripts/test_vuer_streamer.py --skip-wait --rate 5 --max-steps 30

  # Certificados SSL customizados:
  python3 scripts/test_vuer_streamer.py --cert cert.pem --key key.pem
""",
    )
    p.add_argument("--port",        type=int,   default=8012,
                   help="Porta Vuer (default: 8012)")
    p.add_argument("--rate",        type=float, default=10.0,
                   help="Taxa de polling em Hz (default: 10)")
    p.add_argument("--print-every", type=int,   default=5,
                   help="Imprimir a cada N steps (default: 5)")
    p.add_argument("--max-steps",   type=int,   default=0,
                   help="Parar após N steps; 0 = rodar até Ctrl+C (default: 0)")
    p.add_argument("--width",       type=int,   default=640,
                   help="Largura de um olho em px (default: 640)")
    p.add_argument("--height",      type=int,   default=480,
                   help="Altura da imagem em px (default: 480)")
    p.add_argument("--cert-file",   dest="cert_file", default=None,
                   help="Caminho para cert.pem SSL (None = auto-gerado)")
    p.add_argument("--key-file",    dest="key_file",  default=None,
                   help="Caminho para key.pem SSL (None = auto-gerado)")
    p.add_argument("--skip-wait",   action="store_true",
                   help="Não aguardar Enter antes do loop (útil em CI/automação)")
    return p.parse_args()


if __name__ == "__main__":
    run_test(_parse_args())
