"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from typing import ClassVar, TypeVar, Generic, Type

T = TypeVar("T")
_S = TypeVar("_S")

class RegistryMixin(Generic[T]):
    _registry: ClassVar[dict[str, Type[T]]] = {} # type: ignore

    @classmethod
    def register(cls, uid: str):
        def wrapper(subclass: Type[_S]) -> Type[_S]:
            # if not issubclass(subclass, T):
            #     raise TypeError(f"{subclass.__name__} must inherit from {cls._base_type().__name__}")
            cls._registry[uid] = subclass  # type: ignore
            return subclass
        return wrapper

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> T:
        """Construct a fresh instance of whatever class is registered under `uid`.

        NOT memoized -- `make()` used to cache the first-built instance per uid
        (`cls._instances[uid]`) and silently return that same cached object on every later call,
        ignoring any new `*args`/`**kwargs` entirely. This was never noticed because every
        existing call site (every task's `RobotRegistry.make(**self.robot_cfg)` in its own
        `__init__`, `TaskRegistry.make(task, **kwargs)` in `base_dual_env.py`) only ever
        constructs a given uid once per process. It broke as soon as a single long-running
        process constructed the same task/robot uid more than once with different kwargs (e.g.
        an interactive script rebuilding `simple/MissTabletopGraspMP-v0` with a new
        `table_height` each time): confirmed directly -- the second `MissTabletopGraspTaskMP(...)`
        silently returned the *first* instance, with the first instance's own `table_height`
        unchanged, and the cached `Miss()` robot instance kept being reused across separate
        `gym.make()`/`env.close()` cycles while the underlying MjModel/MjData were torn down and
        rebuilt fresh each time -- a plausible direct cause of the segfault that surfaced
        alongside the stale-parameter bug, from the robot object's own state (e.g. curobo's
        lazily-cached kinematics/IK solver) outliving the MuJoCo buffers it was built against.
        """
        if uid not in cls._registry:
            raise ValueError(f"No class registered under uid '{uid}'")

        return cls._registry[uid](*args, **kwargs)

