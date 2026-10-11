# Unity as the operator's display: 3D teleoperation

The teleoperation path used to render a stereo pair on the simulation host,
encode it, and push it to the headset as video. The operator saw a flat screen
pinned to their face, at roughly 100 ms and 500 KB/s, with no parallax: moving
their head did nothing until the next encoded frame arrived.

This path sends **geometry once and poses every step** instead. Unity rebuilds
the MuJoCo scene natively in the headset and renders it at the headset's own
rate, so the operator is inside the scene rather than looking at a picture of
it. The robot is not special in this: it is thirty-odd posed meshes like
everything else, which means no FK on the client and no rig to drift out of
sync with the physics.

The same Unity application also sends the operator's controller poses back, so
one app is both display and input.

---

## Running it

The simulator writes the scene to a directory and serves state over WebRTC; the
geometry itself is fetched over plain HTTP, which you serve yourself.

```bash
python src/simple/cli/teleop_decoupled_wbc.py <env-id> \
    --sim-mode=mujoco --no-headless \
    --unity --unity-teleop --unity-ice-host <this-machine's-IP>
```

Then, in the export directory, once the first episode has written it:

```bash
cd data/unity_scene && python -m http.server 8000
```

Order matters: `data/unity_scene` is written during `env.reset()`, so serving
it before the first run serves an empty directory and Unity reports 404.

### Flags

| Flag | Default | What it is for |
| --- | --- | --- |
| `--unity` | off | Export the scene and open the state channel |
| `--unity-teleop` | off | Also take operator input from Unity. Requires `--unity` |
| `--unity-host` | `0.0.0.0` | Signaling bind address |
| `--unity-port` | `8765` | Signaling port |
| `--unity-export-dir` | `data/unity_scene` | Where `scene.json` and the meshes go |
| `--unity-publish-hz` | `60` | State rate. 0 publishes every tick |
| `--unity-ice-host` | — | Rewrite host ICE candidates to this IP. **Needed over Tailscale or any VPN** |
| `--unity-unordered` | off | Unordered, unreliable state channel. See the transport note below |
| `--unity-wrist-correction` | on | Solve the wrist alignment at calibration. Off falls back to the library's guess |
| `--unity-grip-offset` | — | `"x,y,z"` metres, in controller axes. See calibration below |

`--unity-ice-host` is not optional on a VPN. aiortc advertises the host's local
addresses, which the headset cannot route to; without the rewrite the peer
connection completes and no data ever arrives.

---

## Architecture

Three channels cross the wire, and nothing blocks on anything else.

```
 simulator (Linux)                              Unity (headset)
 ─────────────────                              ───────────────
 scene.json + OBJ meshes  ── HTTP, per episode ──▶  build the scene
 state packets            ── WebRTC, 60 Hz    ──▶  set body transforms
 controller + HMD poses   ◀── WebRTC "tracker" ──   read XR input
```

Removing the state channel freezes the scene in the headset without stalling
the simulator; removing the simulator leaves the last frame on screen. The
dataset render for LeRobot never leaves the host and is untouched by any of
this.

### Static and dynamic

A body is **dynamic** if it or an ancestor carries a degree of freedom, or if
it is a mocap body. Everything else is **static**: welded to the world, the same
in every frame.

Static bodies travel in `scene.json` with their world pose and are placed once.
Only dynamic bodies appear in state packets. A tabletop scene is mostly
furniture, so this is most of the scene — and it is not only thrift. The packet
must survive the path MTU in one piece, and Tailscale's 1280 bytes leaves room
for about 42 bodies. Spending slots on a table is what puts a scene over.

The rule walks the parent chain rather than reading `body_dofnum`, because a
body with no joint of its own still moves when something above it does: every
arm link past the shoulder has zero DOFs and is carried by the joints beneath
it. Mocap bodies are the exception the walk alone gets wrong — no joints, but
written directly into `mjData.mocap_pos` — so they are checked separately.

### The wire format

Protocol version 2. A 20-byte header, then 28 bytes per dynamic body:

```
[4s "SUST"][H version][H flags][I scene_id][I frame][H nbody][H pad]
then nbody × (3 × float32 position, 4 × float32 rotation)
```

Slot *k* is the *k*-th entry of the manifest's `dynamic` list, not body *k*.
That is what the version bump is for: a v1 client reading a v2 packet poses the
wrong bodies while looking entirely plausible, and `scene_id` cannot catch it,
because the same scene can be encoded either way.

`scene_id` is FNV-1a over every body name paired with its packet slot. Both
halves are needed, and each was tried alone first:

- Hashing only the streamed names misses an episode that adds a *static* prop.
  The dynamic list is untouched, so the id holds while the client renders the
  previous episode's geometry — and the poses it receives stay valid, so
  nothing else notices.
- Hashing only the names misses the reverse: the same bodies split differently
  moves every slot without touching a name.

Python's `hash()` is salted per process, which is why FNV-1a is spelled out.

### Transport

The state channel is **ordered and reliable**, which is not what the design
called for and is what was measured.

| Configuration | Result over Tailscale |
| --- | --- |
| Unordered, `maxRetransmits=0` | Only the 20-byte probe arrived |
| Unordered, reliable | Nothing arrived |
| Ordered, reliable | All 60 of 60 |

The packet is currently 1252 bytes for the G1 — above the ~1195 a 1280-byte MTU
leaves — so SCTP fragments it, and an unreliable channel drops fragmented
frames outright. Shrinking the packet below the MTU would make the unordered
configuration viable again; see the deferred list.

---

## Frames

### MuJoCo to Unity

Positions map `(x, y, z) → (x, z, y)` and quaternions `(w, x, y, z) →
(x, z, y, -w)`, which also reverses triangle winding. Verified against the
exporter's own conversion over 300 random orientations, agreeing to 2e-7:

| MuJoCo local axis | Unity local axis |
| --- | --- |
| +x (forward) | **+X** |
| +z (up) | +Y |
| +y (left) | +Z |

Note the trap in the last row: Unity's own `Vector3.forward` is `+Z`, which is
the robot's **left**. Any Unity-side field naming a direction on an exported
body wants `+X` for forward, not `Vector3.forward`.

### The operator's wrists

`UnityTrackerSource.wrist_poses()` applies four steps, in this order:

1. `apply_grip_offset` — move the origin along the controller's own axes
2. `to_robot_frame` — OpenXR (y-up) to the robot's z-up frame
3. `headset_relative_wrist` — subtract the head's position
4. `to_waist_origin` — walk down to the frame the IK solves about

The grip offset must come first, while the pose is still in the controller's
frame. The last two are translations and commute with each other, but neither
commutes with the basis change.

Steps 3 and 4 do not actually steer the robot. `WristsPreProcessor` is
differential — only `inv(wrist_at_calibration) @ wrist_now` reaches the
controller — so a constant translation appears on both sides of that product
and cancels exactly. They are there for parity with `TeleVuerWrapper` and the
xr_teleoperate bridge, so the three can be read side by side.

What survives the differencing is the **rotation**, because the delta is routed
through it. That is where an arm that points the wrong way comes from.

---

## The wrist alignment, solved rather than named

This is the subtlest thing in the path and cost the most to find.

`WristsPreProcessor` maps

```
E = E0 @ inv(A) @ (inv(W0) @ W) @ A
```

where `W0` is the operator's wrist at calibration, `E0` the robot's hand, and
`A` the alignment between the two frames. A hand translation `d` therefore
reaches the robot as `R_e @ A.T @ R_w.T @ d` — routed through the operator's
wrist orientation and applied in the robot's hand frame.

The library picks `A` from the **device name**: identity for `"pico"` and
`"vuer"`, a hardcoded pair for anything else. Both are assertions about what a
particular headset reports. Unity sends `XRNode` device poses, matching
neither, and the symptom was one arm tracking while the other inverted — the
G1's two hand frames are mirror images, so one wrong convention lands
differently on each arm, and no single constant can serve both.

`A` does not need guessing. Setting `R_e @ A.T @ R_w.T = I` gives

```
A = R_w.T @ R_e
```

and both rotations are already stored by `calibrate()`. `wrist_alignment.py`
reads them back and overwrites the guess, on every calibration. Rotation falls
out of the same substitution: a hand rotation `dR` arrives as `dR @ R_e`, which
turns the robot's hand by `dR` in the world frame.

This holds for any device, including ones nobody has written a branch for —
which is the property that makes a mocap suit cheap to add later.

### Confirmed against the real robot, from the other direction

The xr_teleoperate branch `teleop_unity_real` hit the same mismatch and solved
it by measurement: its Unity bridge applies a **fixed 60° pitch** to the wrist
pose, with the note that Unity's pose arrives about that far off, and that
without it the IK reaches the right place while twisting shoulder and elbow.
That is the Meta controller's `XRNode` device pose against OpenXR's grip pose,
measured rather than assumed — the discrepancy this module was written on the
suspicion of.

It also demonstrates the difference between the two approaches. That branch
drives a real G1 through absolute IK, so the offset has to be found and
hardcoded, and a different controller would need it found again. Here it is
absorbed: writing the incoming wrist as `R_w = R_true @ C` for any constant
`C`, the alignment becomes `A = Cᵀ R_trueᵀ R_e`, and the `C` terms cancel in
`Aᵀ (inv(W0) W) A`. Checked numerically over 500 random configurations with
`C` set to that same 60° pitch: the delivered pose is identical to the
no-offset case, and still exactly `dR @ R_e`, to 3e-15.

So this path needs no equivalent of that constant, and would need none for a
mocap suit either.

**Operational consequence:** the alignment is computed from where the
operator's hands were when they pressed activate. Hold a neutral, comfortable
posture at that moment; it becomes the zero for both arms. Deactivate and
reactivate to redo it.

---

## Why the view follows the robot's yaw

`SimpleAvatarView` turns the rig with the robot by default. This is not a
comfort preference — it is what the mapping requires.

The teleoperation IK solves against a model built with
`set_floating_base=False`; nothing in decoupled_wbc passes `True`. The model's
root is therefore the pelvis, and every pose `calibrate()` stores is
pelvis-relative. **The hand targets are in the robot's own base frame.** Turn
the robot and its hands turn with it.

The operator's forward, meanwhile, never changes: `TrackerSender` reads
`InputTracking`, which reports in tracking space and ignores the rig transform
entirely — the property that makes moving the rig safe in the first place. With
the view not turning, the two frames drift apart by exactly the angle the base
has turned since calibration, and the operator pushes their hand one way while
the robot's hand goes another.

Turning physically does not rescue it. The wrist is made relative to the head's
*position* but not its *rotation*, so turning your body swings your hands
around you in tracking space and the robot's arms follow. You cannot turn to
look without moving the robot's arms.

So the arrangement is: the operator keeps facing one physical direction, steers
with the right thumbstick, and the view comes along. Head rotation stays
completely free — the rig turns underneath, the headset's own rotation is
applied on top.

---

## The Unity side

| Component | Job |
| --- | --- |
| `SimpleSceneLoader` | Fetch `scene.json` and the meshes, build the scene |
| `SimpleStateReceiver` | Decode packets, apply poses, count what it refuses |
| `SimpleAvatarView` | Put the viewpoint inside the robot's head |
| `SimpleDesktopXR` | Drive the whole path from keyboard and mouse, no headset |
| `SimpleDebugConnect` | `O` connects, `P` prints status |
| `TrackerSender` | Send controller and HMD poses on the `tracker` channel |

Two fields need setting by hand. Everything else resolves in `Awake`.

- `SimpleSceneLoader.sceneUri` → `http://<sim-host>:8000/scene.json`
- `SimpleAvatarView.xrOrigin` → the XR Origin **root**

The root matters: `XROrigin` rewrites its Camera Offset's local position every
frame to apply the tracking origin mode, so anything written there is undone.

### Changing a default does not change a scene

Serialized component values win over code defaults. Several defaults have moved
during development — `followYaw`, `headForwardAxis`, `headBodyName`,
`eyeOffset` — and a scene saved before a change keeps the old value silently.
No test catches this. After pulling, check the Inspector against the tooltips.

### Testing without a headset

`SimpleDesktopXR` produces the same three poses and button state the real
controllers do, so everything downstream runs unchanged. TrackerSender consults
it only when neither controller reports itself valid, so plugging a headset
back in needs no change.

XRI ships an XR Device Simulator, which is the obvious alternative and does not
work here: it simulates through the Input System while `TrackerSender` reads
the legacy XR API, which a simulated Input System device never registers with.
The camera would move and the robot would receive nothing.

---

## Tests

Four suites need only numpy and run anywhere:

```bash
python scripts/test_unity_streamer.py        # input path, dead zones, toggles
python scripts/test_unity_wrist_alignment.py # the alignment, against mirrored hands
python scripts/test_unity_dynamic_split.py   # which bodies get streamed, MTU budget
```

Three need mujoco and a model, and default to the G1 this checkout ships:

```bash
python scripts/test_unity_export.py
python scripts/test_unity_bridge.py
python scripts/test_unity_webrtc.py
```

The C# side has its own harness in the Unity project under
`Tests~/FormatCompat`: it compiles the real runtime sources against Unity stubs
and decodes a packet Python generated, so the two languages' idea of the layout
cannot drift apart. Fixtures carry a protocol version and must be regenerated
when the format changes — a stale `fixtures/` fails with `unsupported protocol
version N`, which is the check working.

**What no test here can tell you:** whether a wrist convention is right. The
arithmetic is checked and the alignment is checked against mirrored hand
frames, but whether the result is what the robot wants was only ever settled by
moving one.

---

## Known limits and deferred decisions

Written down as decisions rather than left as forgetting.

**The packet does not fit the MTU.** 1252 bytes against ~1195 usable. The
reliable channel absorbs it by fragmenting; the unordered one cannot. Encoding
the quaternion as smallest-three would take a body from 28 to 16 bytes — 724
bytes for the G1 — and reopen the unordered configuration the design wanted.

**Objects render as faceted convex hulls, with no textures.** One root cause:
`MujocoEngine._build_object` adds only `collision_meshes_mujoco`, so the asset's
textured visual mesh never enters the MuJoCo model, and the exporter falls back
to drawing every convex piece. The bridge logs how many geoms came from this
fallback. Fixing it would also cut load time: a warehouse fixture decomposes
into over a thousand pieces.

**No interpolation between state packets.** Publishing at 60 Hz against a
headset rendering at 72–90 means some frames repeat a pose. Interpolating costs
one packet of latency; extrapolating costs overshoot. Which is preferable
depends on what the operator notices, and nobody has looked.

**The grip offset has never been calibrated.** Note this is the *translation*,
and it is a different problem from the 60° rotation that `teleop_unity_real`
fixed — that one this path absorbs, this one it does not. A controller-local
translation right-multiplies the pose, so it conjugates the differential rather
than cancelling in it, and it is still open in both repositories.

The machinery is there and set to zero. To find it: hold the controller still
and rotate it in place; the robot's wrist should turn without translating, and
with no offset applied it does so exactly. Any drift you see is the pivot
error, which makes it measurable by hand — this is the "the wrists feel wrong,
like the pivot is at the base of the controller rather than the middle"
complaint, and nothing has fixed it yet.

**The camera is pitched 47.6° down.** Computed from
`MujocoEngine._build_camera`'s `eye_in_head` mount. Every pixel is below the
horizon, so the robot cannot see its own eye level or above. Fine for
manipulation, limiting for anything else.

---

## Things that cost a session to learn

- `--unity-ice-host` is required on a VPN, and its absence looks like a working
  connection that delivers nothing.
- aiortc's SDP ends with CRLF; Unity's `Split` + `Join` + appended CRLF makes a
  blank line that libwebrtc rejects as "Invalid SDP line". `clean_sdp_for_unity`
  strips it.
- SIMPLE configures no logging, so `logger.info` goes nowhere by default. The
  Unity modules attach their own handler.
- The bridge must be built **after** the first `env.reset()`. `mjModel` does not
  exist until `update_layout()` runs.
- Mesh positions are already compensated in the compiled model. Reading
  `MjModel` rather than source MJCF avoids applying the CoM recentering twice.

---

## See also

- `docs/ARQUITETURA_TELEOP_UNITY.md` and `docs/GUIA_RAPIDO_TELEOP_UNITY.md` on
  the xr_teleoperate branch `teleop_unity_real` — the same Unity client against
  the real G1, including the measured wrist pitch offset discussed above.
- `src/simple/teleop/unity/` — every module carries its reasoning in its
  docstring; this guide is the map, not a replacement for them.
