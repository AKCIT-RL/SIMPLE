"""
Feed the whole-body controller from Unity's XR input.

``TeleopPolicy`` consumes a ``BaseStreamer``; ``PicoStreamer`` backs it with the
XRoboToolkit SDK and ``VuerStreamer`` with a WebXR browser session. Both take
over the headset for themselves, which leaves no XR session for a Unity app --
and so no way to be inside the 3D scene while driving the robot. This backs the
same interface with the ``tracker`` data channel the Unity client already
sends, so one application both renders the scene and steers.

Frames
------
Unity sends raw OpenXR poses as 4x4 column-major matrices, converted from its
own left-handed basis by ``TrackerSender.ConvertMatrix``. Getting from there to
what the controller wants takes two more steps, and they are separable:

``to_robot_frame`` is the basis change, ported from the ``UnityTeleVuerBridge``
in xr_teleoperate, where it drives a real G1. It is arithmetic, and the tests
cover it.

``headset_relative_wrist`` then ``to_waist_origin`` place that pose on the
origin the IK actually solves about. Both are translations and both are
required: the headset reports where the operator's head is, the solver wants a
frame near the robot's waist, and the gap between them is the robot's own
build. ``TeleVuerWrapper`` applies exactly this pair, and so does the
xr_teleoperate bridge that drives a real G1.

Getting these wrong does not look like a frame error. The target lands outside
the arm's reach, the IK returns the least-bad pose it can find, and the arm
tracks the hand loosely while pointing somewhere else -- which reads as an
inverted axis or broken IK. Before concluding that an axis is flipped, check
that the target is reachable at all.
"""

import json
import threading
from dataclasses import dataclass, field

import numpy as np

# OpenXR (y-up, right-handed) to the robot's z-up frame, and back.
T_ROBOT_OPENXR = np.array(
    [
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

T_OPENXR_ROBOT = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

# Head to waist, in the robot frame. The IK solves in a frame whose origin sits
# near the waist joint, but a headset only knows where the head is, so the
# head-relative wrist has to be walked down to that origin before it means
# anything to the solver.
#
# These are the numbers TeleVuerWrapper applies and that the xr_teleoperate
# bridge reproduces to drive a real G1; they are a property of the G1's
# geometry, not of any headset, which is why both paths carry the same pair.
#
# Leaving them out does not tilt the arm, it puts the target roughly at the
# robot's knees and behind it -- outside the arm's reach -- and an IK asked for
# an unreachable pose returns whatever is least bad. The result tracks the hand
# loosely and points somewhere else entirely, which is easy to misread as an
# inverted axis.
WAIST_FROM_HEAD = np.array([0.15, 0.0, 0.45], dtype=np.float64)

# Poses to hold when Unity has sent nothing yet, so the controller reads a
# plausible standing posture instead of a stack of identity matrices at the
# origin. Same values the xr_teleoperate bridge falls back to.
REST_HEAD = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]], dtype=np.float64
)
REST_LEFT_WRIST = np.array(
    [[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]], dtype=np.float64
)
REST_RIGHT_WRIST = np.array(
    [[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1]], dtype=np.float64
)


def matrix_from_payload(values) -> np.ndarray:
    """Read a 4x4 from the 16 column-major floats Unity sends."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 16 values for a matrix, got {arr.size}")
    return arr.reshape((4, 4), order="F")


def is_usable_pose(matrix: np.ndarray) -> bool:
    """Reject a pose that cannot be inverted or composed.

    An XR device reports identity, zeros or NaN while a controller is asleep or
    outside the tracking volume. Feeding those to the IK produces a lunge to the
    origin, so the caller holds the previous pose instead.
    """
    if not np.all(np.isfinite(matrix)):
        return False
    determinant = np.linalg.det(matrix)
    return bool(
        np.isfinite(determinant) and not np.isclose(determinant, 0.0, atol=1e-6)
    )


def to_robot_frame(pose: np.ndarray) -> np.ndarray:
    """Change an OpenXR pose into the robot's frame."""
    return T_ROBOT_OPENXR @ pose @ T_OPENXR_ROBOT


def apply_grip_offset(pose: np.ndarray, offset) -> np.ndarray:
    """Move a controller pose's origin along the controller's own axes.

    A controller reports a pose whose origin sits at some point on its body,
    and runtimes disagree about which point: WebXR's grip space lands near the
    middle of the handle, Unity's XRNode nearer its base. The gap is a few
    centimetres, fixed, and rigid.

    It has to be applied in the controller's frame, not the world's. A
    world-frame shift moves the target by a constant vector no matter how the
    hand is turned, which leaves the pivot in the wrong place: rotating the
    wrist then swings the target through an arc instead of turning it on the
    spot, and the robot's hand orbits. That is what "the wrists feel wrong"
    describes, and no amount of tuning a world-frame offset fixes it, because
    the error depends on orientation.

    Rotation is unaffected. That part was already right: on the xr_teleoperate
    side, roll on the controller produced clean roll on the robot with no
    correction, which is why only the origin needs moving.

    Args:
        pose: 4x4 controller pose, in whatever frame it arrived in.
        offset: 3-vector in the controller's local frame. Calibrate by holding
            the controller still and rotating it in place: the robot's wrist
            should turn without translating. Whichever way it drifts is the
            axis to correct.
    """
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError(f"Expected a 3-vector offset, got {offset.shape}")
    if not offset.any():
        return pose
    shifted = pose.copy()
    shifted[0:3, 3] += pose[0:3, 0:3] @ offset
    return shifted


def headset_relative_wrist(wrist: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Express a wrist pose relative to the head, both already robot-frame.

    Only the translation is made relative. The orientation stays in the world
    frame because the controller's grip pose already arrives close to the URDF
    convention -- confirmed empirically on the xr_teleoperate side, where
    rolling the controller rolls the wrist cleanly with no correction applied.
    """
    relative = wrist.copy()
    relative[0:3, 3] -= head[0:3, 3]
    return relative


def to_waist_origin(wrist: np.ndarray) -> np.ndarray:
    """Re-origin a head-relative wrist onto the frame the IK solves in.

    Kept separate from ``headset_relative_wrist`` because the two answer
    different questions: that one removes the operator's head, this one accounts
    for the robot's build. Only the second changes if the arm is mounted on a
    different torso.
    """
    shifted = wrist.copy()
    shifted[0:3, 3] += WAIST_FROM_HEAD
    return shifted


@dataclass
class UnityTeleData:
    """One frame of Unity XR input, in the shape the streamer consumes.

    Field names follow the ``TeleData`` that ``PicoStreamer`` and
    ``VuerStreamer`` read, so the three stay comparable side by side.
    """

    head_pose: np.ndarray = field(default_factory=lambda: REST_HEAD.copy())
    left_wrist_pose: np.ndarray = field(default_factory=lambda: REST_LEFT_WRIST.copy())
    right_wrist_pose: np.ndarray = field(
        default_factory=lambda: REST_RIGHT_WRIST.copy()
    )

    left_ctrl_trigger: bool = False
    right_ctrl_trigger: bool = False
    left_ctrl_squeezeValue: float = 0.0
    right_ctrl_squeezeValue: float = 0.0

    left_ctrl_aButton: bool = False
    left_ctrl_bButton: bool = False
    right_ctrl_aButton: bool = False
    right_ctrl_bButton: bool = False

    left_ctrl_thumbstick: bool = False
    right_ctrl_thumbstick: bool = False

    left_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    right_ctrl_thumbstickValue: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )


class UnityTrackerSource:
    """Accumulate Unity's ``tracker`` messages into the latest pose and inputs.

    Messages arrive on the WebRTC callback thread and are read by the control
    loop, so the two share a lock. Only the newest frame is kept: this is state,
    not a queue, and a backlog would only be several stale pictures of where the
    operator's hands used to be.
    """

    def __init__(self, grip_offset=(0.0, 0.0, 0.0)) -> None:
        self._lock = threading.Lock()
        self._data = UnityTeleData()
        self.received = 0
        self.rejected = 0
        # Controller-local, so it rotates with the hand. Zero by default: a
        # silent constant here would be indistinguishable from a frame bug.
        self.grip_offset = np.asarray(grip_offset, dtype=np.float64)

    def feed(self, message) -> bool:
        """Ingest one ``tracker`` message. Returns False if it was unusable.

        Safe to pass straight to ``UnityStateServer(on_tracker=...)``; malformed
        input is counted rather than raised, since a dropped input frame is
        recoverable and an exception on the WebRTC thread is not.
        """
        if isinstance(message, (bytes, bytearray)):
            message = message.decode("utf-8", errors="replace")
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            with self._lock:
                self.rejected += 1
            return False
        if not isinstance(payload, dict):
            with self._lock:
                self.rejected += 1
            return False

        with self._lock:
            data = self._data
            for key, attribute in (
                ("head", "head_pose"),
                ("left", "left_wrist_pose"),
                ("right", "right_wrist_pose"),
            ):
                if key not in payload:
                    continue
                try:
                    pose = matrix_from_payload(payload[key])
                except (TypeError, ValueError):
                    continue
                # Hold the last good pose rather than snapping to rest, which
                # would look like the operator flinching every time tracking
                # blinks.
                if is_usable_pose(pose):
                    setattr(data, attribute, pose)

            data.left_ctrl_trigger = float(payload.get("leftTrigger", 0.0)) > 0.5
            data.right_ctrl_trigger = float(payload.get("rightTrigger", 0.0)) > 0.5
            data.left_ctrl_squeezeValue = float(payload.get("leftGrip", 0.0))
            data.right_ctrl_squeezeValue = float(payload.get("rightGrip", 0.0))

            data.left_ctrl_aButton = bool(payload.get("leftPrimary", False))
            data.left_ctrl_bButton = bool(payload.get("leftSecondary", False))
            data.right_ctrl_aButton = bool(payload.get("rightPrimary", False))
            data.right_ctrl_bButton = bool(payload.get("rightSecondary", False))

            data.left_ctrl_thumbstick = bool(payload.get("leftStickClick", False))
            data.right_ctrl_thumbstick = bool(payload.get("rightStickClick", False))

            data.left_ctrl_thumbstickValue = np.array(
                [
                    float(payload.get("leftStickX", 0.0)),
                    float(payload.get("leftStickY", 0.0)),
                ]
            )
            data.right_ctrl_thumbstickValue = np.array(
                [
                    float(payload.get("rightStickX", 0.0)),
                    float(payload.get("rightStickY", 0.0)),
                ]
            )
            self.received += 1
        return True

    def snapshot(self) -> UnityTeleData:
        """A copy of the latest frame, safe to read off the control loop."""
        with self._lock:
            data = self._data
            return UnityTeleData(
                head_pose=data.head_pose.copy(),
                left_wrist_pose=data.left_wrist_pose.copy(),
                right_wrist_pose=data.right_wrist_pose.copy(),
                left_ctrl_trigger=data.left_ctrl_trigger,
                right_ctrl_trigger=data.right_ctrl_trigger,
                left_ctrl_squeezeValue=data.left_ctrl_squeezeValue,
                right_ctrl_squeezeValue=data.right_ctrl_squeezeValue,
                left_ctrl_aButton=data.left_ctrl_aButton,
                left_ctrl_bButton=data.left_ctrl_bButton,
                right_ctrl_aButton=data.right_ctrl_aButton,
                right_ctrl_bButton=data.right_ctrl_bButton,
                left_ctrl_thumbstick=data.left_ctrl_thumbstick,
                right_ctrl_thumbstick=data.right_ctrl_thumbstick,
                left_ctrl_thumbstickValue=data.left_ctrl_thumbstickValue.copy(),
                right_ctrl_thumbstickValue=data.right_ctrl_thumbstickValue.copy(),
            )

    def wrist_poses(self):
        """Both wrists in the robot frame, on the origin the IK solves in.

        Four steps, in this order: move the origin along the controller, change
        basis, subtract the head, then walk down to the waist. The grip offset
        has to come first, while the pose is still in the controller's own
        frame; the last two are translations and commute with each other, but
        neither commutes with the basis change.
        """
        data = self.snapshot()
        head = to_robot_frame(data.head_pose)

        def wrist(raw):
            robot_frame = to_robot_frame(apply_grip_offset(raw, self.grip_offset))
            return to_waist_origin(headset_relative_wrist(robot_frame, head))

        return wrist(data.left_wrist_pose), wrist(data.right_wrist_pose)

    def stats(self) -> dict:
        with self._lock:
            return {"received": self.received, "rejected": self.rejected}


def apply_dead_zone(value: float, dead_zone: float) -> float:
    """Drop small stick deflections and rescale the rest to the full range.

    Without the rescale, the stick would jump to ``dead_zone`` the instant it
    left the centre. Matches PicoStreamer and VuerStreamer exactly, so the three
    feel the same in the hand.
    """
    if abs(value) < dead_zone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)


def finger_data_from_controller(trigger: bool, squeeze: bool) -> np.ndarray:
    """Encode a controller's grip as the (25, 4, 4) array the retargeter reads.

    A controller has no fingers, so trigger and squeeze select which one closes.
    The convention is PicoStreamer's: one entry per fingertip, the flag in
    ``[0, 3]``, 1.0 closed and 0.0 open. The thumb stays open throughout.
    """
    fingertips = np.zeros((25, 4, 4))
    THUMB, INDEX, MIDDLE, RING = 0, 5, 10, 15

    fingertips[4 + THUMB, 0, 3] = 1.0

    if trigger and not squeeze:
        fingertips[4 + INDEX, 0, 3] = 1.0
    elif trigger and squeeze:
        fingertips[4 + MIDDLE, 0, 3] = 1.0
    elif not trigger and squeeze:
        fingertips[4 + RING, 0, 3] = 1.0

    return fingertips


class UnityStreamerCore:
    """Turn Unity input into controller commands, without the SDK dependency.

    Kept free of ``decoupled_wbc`` so the arithmetic here -- dead zones, yaw
    integration, height clamping, edge detection -- is testable on a machine
    that cannot install the controller stack. ``make_unity_streamer`` wraps this
    in the ``BaseStreamer`` the policy expects.

    Constants mirror PicoStreamer and VuerStreamer; changing one here alone
    would make Unity feel different from the other two for no stated reason.
    """

    DEAD_ZONE = 0.1
    MAX_LINEAR_VEL = 0.5  # m/s
    MAX_ANGULAR_VEL = 1.0  # rad/s
    CONTROL_DT = 1.0 / 50.0  # the policy calls get() at 50 Hz

    # Sign per stick axis, as Unity reports them.
    #
    # VuerStreamer negates the forward axis because WebXR's gamepad spec makes
    # y positive when the stick is pulled *towards* the operator. Unity's
    # primary2DAxis is positive away from them, so copying that negation sends
    # the robot backwards. The other two axes agree between the runtimes.
    #
    # Flip one of these if a runtime update changes a convention; the symptom
    # is an axis that drives the opposite way, and nothing else.
    FORWARD_SIGN = 1.0
    STRAFE_SIGN = -1.0
    YAW_SIGN = -1.0

    HEIGHT_INCREMENT = 0.01
    HEIGHT_MIN = 0.20
    HEIGHT_MAX = 0.74
    HEIGHT_DEFAULT = 0.74

    def __init__(self, source: UnityTrackerSource) -> None:
        self.source = source
        self.reset_status()

    def reset_status(self) -> None:
        self.current_base_height = self.HEIGHT_DEFAULT
        self.target_yaw = 0.0
        self._activation_last = False
        self._policy_last = False
        self._collection_last = False
        self._abort_last = False

    @staticmethod
    def _rising_edge(now: bool, previous: bool) -> bool:
        """True only on the False->True transition.

        Every toggle is edge-triggered: held buttons are read 50 times a second,
        and a level-triggered toggle would flip on each of them.
        """
        return bool(now) and not previous

    def poll(self) -> dict:
        """One frame of commands, in the shape ``StreamerOutput`` takes."""
        data = self.source.snapshot()
        left_wrist, right_wrist = self.source.wrist_poses()

        # Left stick drives the base, right stick turns it. See the sign
        # constants above for why forward is not negated the way VuerStreamer
        # negates it.
        forward = self.FORWARD_SIGN * float(data.left_ctrl_thumbstickValue[1])
        strafe = self.STRAFE_SIGN * float(data.left_ctrl_thumbstickValue[0])
        yaw_rate_in = self.YAW_SIGN * float(data.right_ctrl_thumbstickValue[0])

        lin_vel_x = apply_dead_zone(forward, self.DEAD_ZONE) * self.MAX_LINEAR_VEL
        lin_vel_y = apply_dead_zone(strafe, self.DEAD_ZONE) * self.MAX_LINEAR_VEL
        vyaw = apply_dead_zone(yaw_rate_in, self.DEAD_ZONE) * self.MAX_ANGULAR_VEL

        # Yaw is a heading, so the rate is integrated and wrapped rather than
        # passed through. Left unwrapped it grows without bound and the angle
        # loses precision after a few minutes of turning.
        self.target_yaw += vyaw * self.CONTROL_DT
        self.target_yaw = float(
            np.arctan2(np.sin(self.target_yaw), np.cos(self.target_yaw))
        )

        if data.right_ctrl_aButton:
            self.current_base_height += self.HEIGHT_INCREMENT
        elif data.right_ctrl_bButton:
            self.current_base_height -= self.HEIGHT_INCREMENT
        self.current_base_height = float(
            np.clip(self.current_base_height, self.HEIGHT_MIN, self.HEIGHT_MAX)
        )

        toggle_activation = self._rising_edge(
            data.left_ctrl_bButton, self._activation_last
        )
        self._activation_last = bool(data.left_ctrl_bButton)

        toggle_policy = self._rising_edge(data.left_ctrl_aButton, self._policy_last)
        self._policy_last = bool(data.left_ctrl_aButton)

        toggle_collection = self._rising_edge(
            data.right_ctrl_trigger, self._collection_last
        )
        self._collection_last = bool(data.right_ctrl_trigger)

        abort_now = data.right_ctrl_squeezeValue > 0.5
        toggle_abort = self._rising_edge(abort_now, self._abort_last)
        self._abort_last = abort_now

        return {
            "ik_data": {
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
                "left_fingers": {
                    "position": finger_data_from_controller(
                        data.left_ctrl_trigger, data.left_ctrl_squeezeValue > 0.5
                    )
                },
                "right_fingers": {
                    "position": finger_data_from_controller(
                        data.right_ctrl_trigger, data.right_ctrl_squeezeValue > 0.5
                    )
                },
            },
            "control_data": {
                "base_height_command": self.current_base_height,
                "navigate_cmd": [lin_vel_x, lin_vel_y, vyaw, self.target_yaw],
                "toggle_policy_action": toggle_policy,
            },
            "teleop_data": {"toggle_activation": toggle_activation},
            "data_collection_data": {
                "toggle_data_collection": toggle_collection,
                "toggle_data_abort": toggle_abort,
            },
            "source": "unity",
        }


def make_unity_streamer(source: UnityTrackerSource):
    """Build the ``BaseStreamer`` the teleop policy expects.

    ``BaseStreamer`` is imported here rather than at module scope so the rest of
    this file stays usable -- and testable -- without the controller stack
    installed.
    """
    from decoupled_wbc.control.teleop.streamers.base_streamer import (
        BaseStreamer,
        StreamerOutput,
    )

    class UnityStreamer(BaseStreamer):
        """Streamer backed by the Unity client's ``tracker`` data channel."""

        def __init__(self, tracker_source: UnityTrackerSource) -> None:
            super().__init__()
            self._core = UnityStreamerCore(tracker_source)

        # The Unity app owns its own lifecycle; the channel is opened by the
        # signaling server and outlives any single episode.
        def start_streaming(self) -> None:
            pass

        def stop_streaming(self) -> None:
            pass

        def reset_status(self) -> None:
            self._core.reset_status()

        @property
        def current_base_height(self) -> float:
            return self._core.current_base_height

        @property
        def target_yaw(self) -> float:
            return self._core.target_yaw

        def get(self) -> StreamerOutput:
            return StreamerOutput(**self._core.poll())

    return UnityStreamer(source)


class UnityButtonPoller:
    """Edge-detect the buttons SIMPLE handles rather than the controller.

    Dropping the elastic band and resetting the environment are simulator
    concerns, not teleoperation ones, so they stay out of the streamer: the
    policy has no business knowing an elastic band exists. ``PicoDecoupledAgent``
    and ``VuerDecoupledAgent`` both keep them at agent level for the same
    reason, and this mirrors their bindings.

    Bindings, matching VuerDecoupledAgent exactly:
        drop robot   right thumbstick click
        reset env    both grips held together

    Both are edge-triggered. At 50 Hz a level-triggered reset would fire fifty
    times while the operator's hands are still closing.
    """

    def __init__(self, source: UnityTrackerSource, grip_threshold: float = 0.5) -> None:
        self._source = source
        self._grip_threshold = grip_threshold
        self._drop_last = False
        self._reset_last = False
        self.drop_requested = False
        self.reset_requested = False

    def poll(self) -> dict:
        """Read the buttons once. Returns the edges seen this call."""
        data = self._source.snapshot()

        drop_now = bool(data.right_ctrl_thumbstick)
        drop_edge = drop_now and not self._drop_last
        self._drop_last = drop_now

        reset_now = (
            data.left_ctrl_squeezeValue > self._grip_threshold
            and data.right_ctrl_squeezeValue > self._grip_threshold
        )
        reset_edge = reset_now and not self._reset_last
        self._reset_last = reset_now

        self.drop_requested = self.drop_requested or drop_edge
        self.reset_requested = self.reset_requested or reset_edge
        return {"drop": drop_edge, "reset": reset_edge}

    def take_reset_request(self) -> bool:
        """Consume a pending reset. True at most once per press.

        The sim loop calls this where it would check ``agent.reset_requested``;
        clearing on read keeps a single press from resetting twice.
        """
        pending, self.reset_requested = self.reset_requested, False
        return pending

    def take_drop_request(self) -> bool:
        """Consume a pending drop, on the same read-and-clear contract."""
        pending, self.drop_requested = self.drop_requested, False
        return pending

    def reset_status(self) -> None:
        self._drop_last = False
        self._reset_last = False
        self.drop_requested = False
        self.reset_requested = False


def attach_unity_streamer(teleop_policy, source: UnityTrackerSource):
    """Point an existing ``TeleopPolicy`` at Unity's input.

    ``TeleopStreamer`` leaves ``body_streamer`` as None for a device name it
    does not recognise, which is the hook both PicoDecoupledAgent and
    VuerDecoupledAgent use: construct the policy with an unknown device, then
    assign the streamer over the top. Passing "unity" also avoids
    ``DummyStreamer``, which pulls in ROS 2.

    Hands and body share one streamer because one pair of controllers reports
    both, exactly as the Vuer path does.

    Args:
        teleop_policy: a ``TeleopPolicy`` built with
            ``body_control_device="unity"`` and ``hand_control_device="unity"``.
        source: the ``UnityTrackerSource`` the WebRTC channel feeds.

    Returns:
        The streamer that was attached, for ``reset_status()`` on episode reset.
    """
    streamer = make_unity_streamer(source)
    streamer.start_streaming()
    teleop_policy.teleop_streamer.body_streamer = streamer
    teleop_policy.teleop_streamer.hand_streamer = streamer
    return streamer
