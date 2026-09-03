"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Any, List

if TYPE_CHECKING:
    # from simple.core.asset import Asset
    # from simple.core.actor import Actor
    from simple.core.task import Task
    from simple.core.actor import RobotActor, ObjectActor, CameraEntity, ArticulatedObjectActor, StaticObjectActor
    from simple.assets.primitive import Primitive # , Box
    # from simple.sensors.config import CameraCfg
    

import numpy as np
from simple.core.simulator import Simulator

from simple.core.object import SemanticAnnotated # , Object, SpatialAnnotated
# from simple.core.robot import Robot
from simple.robots.protocols import Controllable
from simple.utils import resolve_data_path
# from simple.assets.objects import ObjectsAsset
# from simple.assets.graspnet import Graspnet1BAsset
# from dm_control import mjcf
import mujoco

from PIL import Image
import os
import cv2
import transforms3d as t3d

xyzw_to_wxyz = lambda q: np.array([q[3], q[0], q[1], q[2]])

class MujocoSimulator(Simulator):

    def __init__(self, task:Task, render_hz:int=30, physics_dt: float = 0.002, headless=True) -> None:
        self.task = task
        self.render_hz = task.metadata["render_hz"] if "render_hz" in self.task.metadata else render_hz
        self.physics_dt = task.metadata["physics_dt"] if "physics_dt" in self.task.metadata else physics_dt

        # TODO move to reset? 
        # self.mj_physics = None
        self.mj_worldbody = None
        self.robot_mjcf = None
        self.render_option = None
        self.renderers = {}
        self.default_camera_name = None
        self.render_step = 0
        self.last_action = [0 for _ in range(17)]#TODO

        self.viewer = None
        self.headless=headless
        
        self.need_gravity = self.task.metadata.get("need_gravity", False)
        self._is_sonic = None
        self.articulated_object_joints = None

    def update_layout(self, **kwargs) -> None:
        # FIXME
        self._is_sonic = ("sonic_config" in kwargs)
        self._setup_scene(**kwargs)

    def step(self, render=True, render_robot_mask=False, **kwargs) -> Dict[str, np.ndarray] | None:
        if self.need_gravity:
            if self.task.robot.command is None:# probably resetting?
                mujoco.mj_step(self.mjModel, self.mjData, nstep=1)
            else:
                self.task.robot.step(self.task.robot.command, replay=self.task.robot.is_replay ,eval = self.task.robot.is_eval)
        else:
            timestep = self.mjModel.opt.timestep
            current_physics_time = self.mjData.time
            num_physics_steps = int(((self.render_step + 1) / self.render_hz - current_physics_time) // timestep)
            assert num_physics_steps > 0 and num_physics_steps < 100000, "warning: why so many physics steps?"
            mujoco.mj_step(self.mjModel, self.mjData, nstep=num_physics_steps)

        self.render_step += 1
        if self.viewer is not None:
            self.viewer.sync()

        # updata self.task.layout.objects pose, at beginning few steps don't update
        if self.render_step > 5:
            for objtype , mj_obj in self.mj_objects.items():
                self.task.layout.actors[objtype].pose.position = list(mj_obj.xpos)
                self.task.layout.actors[objtype].pose.quaternion = list(mj_obj.xquat)
            self.task.layout.actors["robot"].pose.position = list(np.round(self.mjData.qpos[:3], 3))
            self.task.layout.actors["robot"].pose.quaternion = list(np.round(self.mjData.qpos[3:7], 3))

        if render:
            return self.render(render_robot_mask)
    
    def set_states(self, states: Dict[str, Any]) -> None: 
        raise NotImplementedError

    def get_states(self) -> Dict[str, Any]:
        obj_positions = [mj_obj.xpos for mj_obj in self.mj_objects.values()]
        obj_orientations = [mj_obj.xquat for mj_obj in self.mj_objects.values()]
        # joint_state = np.array([joint.qpos[0] for joint in self.joints])
        joint_state = self.task.robot.get_robot_qpos()
        robot_position = np.round(self.mjData.qpos[:7], 4)
        # Only real articulated objects expose `articulate_*` joints/bodies.
        # Static furniture also rides the articulated actor path (0 joints), which
        # leaves this list EMPTY but not None — the old `is not None` check then
        # fell through to `mjData.body("articulate_base")` and raised
        # "Invalid name 'articulate_base'". Same guard as _setup_scene.
        if self.articulated_object_joints:
            articulated_joints_state = {}
            for articulate_joint in self.articulated_object_joints:
                # articulated_joints_state[articulate_joint] = self.mjData.joint(articulate_joint).qpos[0]
                raw_qpos = self.mjData.joint(articulate_joint).qpos[0]


                wrapped_qpos = (raw_qpos + np.pi) % (2 * np.pi) - np.pi

                articulated_joints_state[articulate_joint] = wrapped_qpos
            try:
                articulate_object_position = np.round(self.mjData.joint("articulate_floating_base").qpos, 4)
            except (KeyError, ValueError):
        
                body_ptr = self.mjData.body("articulate_base")
                combined_pos = np.concatenate([body_ptr.xpos, body_ptr.xquat])
                articulate_object_position = np.round(combined_pos, 4)
            return [self.render_step, joint_state, self.obj_names, obj_positions, obj_orientations, robot_position, articulated_joints_state, articulate_object_position]
       
            
        else:
            return [self.render_step, joint_state, self.obj_names, obj_positions, obj_orientations, robot_position]


    def _setup_scene(self, **kwargs):
        from simple.core.actor import RobotActor, ObjectActor, ArticulatedObjectActor, CameraEntity, StaticObjectActor
        from simple.assets.primitive import Primitive

        # https://mujoco.readthedocs.io/en/stable/computation/index.html
        # https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip
        mjSpec = mujoco.MjSpec()
        mjSpec.option.timestep = self.physics_dt
        mjSpec.option.impratio = 10
        mjSpec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        mjSpec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        mjSpec.option.noslip_iterations = 2
        # Arena for contacts/constraints. MuJoCo's automatic estimate is sized
        # from a heuristic and overflows ("Insufficient arena memory for the
        # number of constraints generated" -> segfault) on scenes that combine
        # mesh-collision furniture with many convex-hull objects (e.g. the
        # industrial sorting task: 3 furniture pieces + 2 totes + up to 8 parts,
        # 16 hulls each). 256 MB is ample and costs only address space.
        mjSpec.memory = 256 * 1024 * 1024
        
        mj_worldbody = mjSpec.worldbody

        # Body/mesh/geom names default to the asset's label/uid for readability
        # and backward compatibility with tooling that expects e.g. "bin_b04".
        # That collides when multiple instances of the *same* asset are placed
        # by the same task (target_group/shelf_group, e.g. several "toteweg"
        # instances) -- only those duplicated labels get an "_{objtype}" suffix
        # (objtype is the actor's layout key, e.g. "target_0", always unique),
        # so every pre-existing single-instance task keeps its exact prior
        # naming untouched.
        from collections import Counter
        label_counts = Counter()
        for _objtype, _actor in self.task.layout.actors.items():
            if isinstance(_actor, (ObjectActor, StaticObjectActor)):
                _label = _actor.asset.uid
                if isinstance(_actor.asset, SemanticAnnotated):
                    _label = _actor.asset.label
                label_counts[_label] += 1
        self._dup_object_labels = {lbl for lbl, count in label_counts.items() if count > 1}

        mj_worldbody.add_light(
            # type="directional_light",
            pos=[0, 0, 1.5], 
            dir=[0, 0, -1],
            # directional=True, 
            castshadow=False,
            # ambient=1.5
        )

        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, StaticObjectActor):
                self._build_static_object(mjSpec, mj_worldbody, actor, objtype)
            elif isinstance(actor, ObjectActor):
                # This is a string, which means it's a name of an asset
                # asset = self.task.layout.assets[actor]
                # actor = Actor.from_asset(asset)
                self._build_object(mjSpec, mj_worldbody, actor, objtype)
            elif isinstance(actor, RobotActor):
                self._build_robot(mjSpec, mj_worldbody, actor) # HACK make sure called first
            elif isinstance(actor, Primitive):
                self._build_primitive(mjSpec, mj_worldbody, actor, table_name=objtype)
            elif isinstance(actor, ArticulatedObjectActor):
                self._build_articulated_object(mjSpec, mj_worldbody, actor)
            else:
                raise TypeError(f"Unsupported actor type: {type(actor)}")
            
        self.mj_worldbody = mj_worldbody

        for cname, camera in self.task.layout.cameras.items():
            self._build_camera(cname, camera)

        if not self._is_sonic:
            if getattr(self.task.robot, "floor_grounded", False):
                # A "floor_grounded" robot is one whose task deliberately places it so its own
                # floor-contact point (wheels, feet, ...) touches literal world Z=0 -- for Miss,
                # `self.robot_z` (its *origin*, i.e. base_link) is actually `_ROBOT_Z_OFFSET`
                # above that contact point on purpose (see miss_tabletop_grasp_mp.py), so the
                # groundplane belongs at world Z=0 itself, not at `self.robot_z`. The table-height
                # -relative formula below has no relationship to where this robot's floor contact
                # actually is, and -- unlike Vega1's `z_offset` -- cannot be fixed by a constant
                # shift, because its error term scales with table_height, a per-task-instance
                # value. Skip it entirely.
                z_minus = 0.0
            else:
                z_minus = self.task.layout.scene.table.pose.position[2] + 0.5 * self.task.layout.scene.table.size[2]
                if hasattr(self.task.robot, "z_offset"):
                    # HACK for vega robot base height
                    z_minus += self.task.robot.z_offset-self.robot_z
        else:
            z_minus = 0.0

        # The "groundplane" material referenced below has never actually been defined by SIMPLE
        # itself -- every robot MJCF integrated so far (panda.xml, aloha.xml, vega.xml, the G1
        # variants) happens to bundle its own leftover-from-demo-authoring texture/material named
        # exactly "groundplane", which gets pulled in incidentally when the robot body is
        # attached, and this ground geom just opportunistically reuses it. WidowX AI's
        # wxai_follower.xml does not define one, exposing the accidental dependency (compile
        # error: "material 'groundplane' not found"). Define it explicitly and unconditionally
        # here instead (guarding against the also-common case where the robot MJCF already
        # defines one, to avoid a duplicate-name compile error) -- values match panda.xml's own
        # groundplane texture/material 1:1, so this is a no-op visually for existing robots.
        if mjSpec.material("groundplane") is None:
            groundplane_tex = mjSpec.add_texture(
                name="groundplane",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                mark=mujoco.mjtMark.mjMARK_EDGE,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
                markrgb=[0.8, 0.8, 0.8],
                width=300,
                height=300,
            )
            groundplane_mat = mjSpec.add_material(name="groundplane", texuniform=True, reflectance=0.2)
            groundplane_mat.texrepeat = [5, 5]
            groundplane_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"

        # add ground plane
        ground = mj_worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,  # type: ignore
            name="ground",
            size=[0, 0, 1],
            pos=[0, 0, -z_minus],
            material="groundplane"
        )
        # add some friction for contact stability
        ground.friction = [1.0, 0.005, 0.0001]  # [sliding, torsional, rolling]

        # 5. disable gravity (for better PID control of arms)
        self.mjModel=mjSpec.compile()
        self.mjData=mujoco.MjData(self.mjModel)
        self.mjSpec=mjSpec

        if not self.need_gravity:
            self.mjModel.opt.gravity= (0,0,0) # type: ignore
            # physics = mjcf.Physics.from_mjcf_model(mjcf_model)
            # physics.model.opt.gravity= (0,0,0) # type: ignore
            pseudo_gravity = np.zeros(6)
            pseudo_gravity[2] = -9.81 * 0.1#0.1 # 0.1 is the mass of the object. When changing the mass, this value should be changed accordingly!
        else:
            self.mjModel.opt.gravity = (0,0,-9.81)
            pseudo_gravity = np.zeros(6)
            pseudo_gravity[2] = -9.81 * 0.1

        mj_objects = {}
        obj_names = []
        for objtype, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                label = actor.asset.uid
                name = actor.asset.uid
                uid = actor.asset.uid
                if isinstance(actor.asset, SemanticAnnotated):
                    label = actor.asset.label
                    name = actor.asset.name
                    uid = actor.asset.uid
                # mj_obj=physics.data.body(f"{label}")
                # mj_obj.xfrc_applied = pseudo_gravity
                body_name = f"{label}_{objtype}" if label in self._dup_object_labels else label
                mj_obj=self.mjData.body(body_name)
                if objtype == "container":
                    gravity= pseudo_gravity.copy()
                    gravity[2] = -9.81*1
                    mj_obj.xfrc_applied = gravity
                else:
                    mj_obj.xfrc_applied = pseudo_gravity
                mj_objects[objtype] = mj_obj
                # Disambiguated body_name, not the bare label -- consumed by
                # IsaacSimSimulator.sync_states, which zips this against
                # obj_positions/obj_orientations and does
                # self.objects[obj_name] to pick which Isaac prim to move.
                # With several same-asset instances (e.g. multiple bin_b04
                # totes), the bare label collides across all of them; Isaac's
                # object keys are now disambiguated the same way (see
                # update_layout's dup_labels), so this must match.
                obj_names.append(body_name)

        self.mj_objects = mj_objects
        self.obj_names = obj_names

        # if isinstance(self.task.robot, Controllable):
        #     self.task.robot.initialize_controller(data)

        # 6. setup control
        assert isinstance(self.task.robot, Controllable), "Task robot is None."
        self.joints, self.actuators=self.task.robot.setup_control(self.mjData, self.mjModel, mjSpec=self.mjSpec)
        
        
        # Only initialise articulate joints when the scene actually has an
        # articulated object with joints under the canonical "articulated" key.
        # Static furniture is also attached via the articulated path (0 joints,
        # keyed "furniture_*"), so guard against assuming an "articulated" actor.
        if self.articulated_object_joints and "articulated" in self.task.layout.actors:
            articulate_joint_qpos = self.task.layout.actors["articulated"].asset.articulate_init_joint_qpos
            if articulate_joint_qpos is not None:
                for joint_name, qpos in articulate_joint_qpos.items():
                    self.mjData.joint(joint_name).qpos[0] = qpos



        

        # CHECK
        mujoco.mj_forward(self.mjModel, self.mjData)

        # 8. rendering
        self.render_option = mujoco.MjvOption() # type: ignore
        mujoco.mjv_defaultOption(self.render_option) # type: ignore
        # self.render_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1 # type: ignore

        # in case of forgetting to close the env before reset
        if (hasattr(self, "renderers") and len(self.renderers) > 0):
            self.close()

        self.renderers = {}
        for cname, camera in self.task.layout.cameras.items():
            self.renderers[cname] = mujoco.Renderer(
                self.mjModel,
                height=camera.resolution[1],
                width=camera.resolution[0]
            ) # type: ignore

        if not self._is_sonic:
            if self.viewer is not None:
                self.viewer.close()

            if not self.headless:
                # This will display the int running physics
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.mjModel, self.mjData)
            
        # ?. reset render step
        self.render_step = 0

    def _build_object(self, mjSpec, mjWorld, actor: ObjectActor, objtype: str):
        # asset_id = actor.asset.uid

        # TODO primitive types
        collision_meshes = actor.asset.collision_meshes_mujoco
        num_convex = len(collision_meshes)

        label = actor.asset.uid
        name = actor.asset.uid
        if isinstance(actor.asset, SemanticAnnotated):
            label = actor.asset.label
            name = actor.asset.name

        body_name = f"{label}_{objtype}" if label in self._dup_object_labels else label

        for i in range(num_convex):
            mjSpec.add_mesh(
                name=f'{body_name}_mesh_convex{i}',
                file=collision_meshes[i],
            )

        mj_obj=mjWorld.add_body(
            name=body_name,
            pos=actor.pose.position,
            quat=actor.pose.quaternion)


        num_convex = len(collision_meshes)
        for i in range(num_convex):
            mj_obj.add_geom(
                name=f"{body_name}_convex_{i}",
                meshname=f"{body_name}_mesh_convex{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                # opposing slip in the tangent plane, rotation around the contact normal
                # and rotation around the two axes of the tangent plane
                condim=4,
                # total mass 0.1 helps preventing slipping
                mass=0.1/num_convex,
                # rubber on rough ground: large static, sliding and torisonal friction
                friction=[0.8, 0.05, 0.005],
                # Per-actor / per-asset tint, default plain white. actor.rgba is
                # the weg target-tote highlight (set per episode on the one tote
                # the robot must deliver); actor.asset.rgba is a color variant
                # baked into the asset (e.g. bin_b04_red / bin_b04_blue in the
                # sorting task). The two never apply to the same object, so try
                # the actor override first, then the asset variant.
                rgba=(
                    getattr(actor, "rgba", None)
                    or getattr(actor.asset, "rgba", None)
                    or [1, 1, 1, 1]
                ),
                # Padding/placeholder instances opt out of collision entirely.
                # They are parked below the floor to keep a constant object count
                # in the recorded observations, but the ground plane is an
                # INFINITE half-space, so a colliding body parked under it is
                # deeply penetrating and gets ejected upward at ~140 m/s straight
                # through the workspace.
                contype=0 if getattr(actor.asset, "no_collision", False) else 1,
                conaffinity=0 if getattr(actor.asset, "no_collision", False) else 1,
                # Optional contact-detection margin. Set margin==gap so near
                # contacts (up to `contact_margin`) are *reported* in mjData.contact
                # without producing any force (the gap zone is force-free), leaving
                # the physics identical. Used so a tote resting a couple mm above a
                # shelf's convex-hull collision surface still registers as "on the
                # shelf" for the reward/success predicate. Default 0.0 => unchanged.
                margin=getattr(actor.asset, "contact_margin", 0.0),
                gap=getattr(actor.asset, "contact_margin", 0.0),
                # stiff contact and no oscillation
                solref = [0.005, 2]
            )
        mj_obj.add_freejoint(name=f'{body_name}_joint')

    def _build_static_object(self, mjSpec, mjWorld, actor: StaticObjectActor, objtype: str):
        """Identical to _build_object, except the body is left fixed to the
        world (no free joint) -- for immovable scenario geometry that still
        needs real MuJoCo collision, e.g. a shelf unit."""
        collision_meshes = actor.asset.collision_meshes_mujoco
        num_convex = len(collision_meshes)

        label = actor.asset.uid
        name = actor.asset.uid
        if isinstance(actor.asset, SemanticAnnotated):
            label = actor.asset.label
            name = actor.asset.name

        body_name = f"{label}_{objtype}" if label in self._dup_object_labels else label

        for i in range(num_convex):
            mjSpec.add_mesh(
                name=f'{body_name}_mesh_convex{i}',
                file=collision_meshes[i],
            )

        mj_obj = mjWorld.add_body(
            name=body_name,
            pos=actor.pose.position,
            quat=actor.pose.quaternion)

        for i in range(num_convex):
            mj_obj.add_geom(
                name=f"{body_name}_convex_{i}",
                meshname=f"{body_name}_mesh_convex{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                condim=4,
                friction=[0.8, 0.05, 0.005],
                rgba=[1, 1, 1, 1],
                solref=[0.005, 2],
            )
        # deliberately no add_freejoint(): this body stays welded to the world.

    def _build_articulated_object(self, mjSpec, mjWorld, actor: ArticulatedObjectActor):
        """Build the articulated object in the Mujoco simulator."""
        articulated_object_mjcf = mujoco.MjSpec.from_file(resolve_data_path(actor.asset.mjcf_path, auto_download=True))
        # for geom in articulated_object_mjcf.geoms:
        #     geom.density = 30.0

        geoms = articulated_object_mjcf.geoms
        num_geoms = len(geoms)
        
        # if num_geoms > 0:
           
        #     target_total_mass = 2  
        #     mass_per_geom = target_total_mass / num_geoms
            
        #     for geom in geoms:
        #         geom.mass = mass_per_geom

        self.articulated_object_joints = []
        

        for joint in articulated_object_mjcf.joints:
            if joint.name and joint.name.startswith("articulate_joint"):
                self.articulated_object_joints.append(joint.name)


        frame = mjWorld.add_frame(pos=actor.pose.position, quat=actor.pose.quaternion)
        mjSpec.attach(articulated_object_mjcf, frame=frame)
        self.articulated_object_mjcf = articulated_object_mjcf

    # def _build_articulated_object(self, mjSpec, mjWorld, actor: ArticulatedObjectActor):
    #     """Build the articulated object in the Mujoco simulator with volume-proportional mass."""
    #     articulated_object_mjcf = mujoco.MjSpec.from_file(resolve_data_path(actor.asset.mjcf_path, auto_download=True))
        
    #     # --- Step 1: Perform a "dummy compile" with a base density to probe the total volume ---
    #     base_density = 1000.0  # Assume an initial density of 1000 kg/m^3 (density of water)
    #     for geom in articulated_object_mjcf.geoms:
    #         # CRITICAL: mass must be strictly 0.0 so MuJoCo calculates mass using density * volume.
    #         # Note: Ensure your XML has <compiler boundmass="0.001" boundinertia="0.000001" /> 
    #         # to prevent crashes on zero-volume geoms.
    #         geom.mass = 0.0  
    #         geom.density = base_density
            
    #     # Compile a temporary model. This is very fast and forces MuJoCo to compute 
    #     # the exact volumes, inertias, and masses under the hood.
    #     temp_model = articulated_object_mjcf.compile()
        
    #     # Get the current total mass calculated using the base_density
    #     current_total_mass = sum(temp_model.body_mass)
        
    #     # --- Step 2: Calculate the true density required to hit the target mass ---
    #     target_total_mass = 2.0  # The strict total mass you want for the object
        
    #     if current_total_mass > 0:
    #         # Scale factor = Target Mass / Current Mass
    #         density_scale = target_total_mass / current_total_mass
    #         target_density = base_density * density_scale
            
    #         # --- Step 3: Apply the correct density back to all geometries ---
    #         for geom in articulated_object_mjcf.geoms:
    #             geom.mass = 0.0             # Ensure mass is still 0.0 to force density usage
    #             geom.density = target_density

    #     # Finally, attach the perfectly balanced object to the main world frame
    #     frame = mjWorld.add_frame(pos=actor.pose.position, quat=actor.pose.quaternion)
    #     mjSpec.attach(articulated_object_mjcf, frame=frame)
        
    #     self.articulated_object_mjcf = articulated_object_mjcf


        
      





    def _build_robot(self, mjSpec, mjWorld, actor: RobotActor):
        """Build the robot in the Mujoco simulator."""
        # try: 
        # 1. resolve some local file path which is need to run the script
        robot_mjcf=mujoco.MjSpec.from_file(resolve_data_path(actor.robot.mjcf_path, auto_download=True))
        # except FileNotFoundError:
        #     # 2. catch file not found error if files are not auto-downloaded
        #     from huggingface_hub import snapshot_download
        #     local_data_dir =self.resolve_data_path()
        #     # robot=actor.robot.mjcf_path.split('/')[-2]
        #     print(f"Auto downloading assets to {local_data_dir} ...")
        #     # # 3. Now auto download it using huggingface-hub
        #     snapshot_download(
        #         repo_id="SIMPLE-org/SIMPLE",
        #         allow_patterns=["robots.zip"],
        #         local_dir=local_data_dir,
        #         repo_type="dataset",
        #         # resume_download=True,
        #         token="YOUR_HF_TOKEN",
        #     )
        #     # 4. unzip the downloaded zip file
        #     import zipfile
        #     zip_path = os.path.join(local_data_dir, "robots.zip")
        #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #         zip_ref.extractall(local_data_dir)
        #     if os.path.exists(zip_path):
        #         os.remove(zip_path)
        #         print(f"Deleted {zip_path}")

        #     robot_mjcf=mujoco.MjSpec.from_file(self.resolve_data_path(actor.robot.mjcf_path))

        for g in robot_mjcf.geoms:
            if "hand" in (g.name or g.meshname) or "gripper" in (g.name or g.meshname):
                # This solimp and solref comes from the Shadow Hand xml
                # They can generate larger force with smaller penetration
                # The body will be more "rigid" and less "soft"
                g.solimp[:3] = [0.9, 0.99, 0.0001]
                g.solref[:2] = [0.005, 1]

        self.robot_z = actor.pose.position[2]
        frame=mjWorld.add_frame(pos=actor.pose.position,quat=actor.pose.quaternion) 
        mjSpec.attach(robot_mjcf,frame=frame)

        self.robot_mjcf = robot_mjcf

    def _build_primitive(self, mjSpec, mjWorld, actor: Primitive, table_name: str = 'table'):
        from simple.assets.primitive import Box
        if isinstance(actor, Box):
            table_size = 0.5 * np.array(actor.size)
            table_position = actor.pose.position.copy()
            if not self._is_sonic and table_name == 'table': # allow personalization of other tables
                table_z = - 0.5 * actor.size[2] 
                table_position[2] = table_z 
            
            table=mjWorld.add_body(name=table_name,
                                   pos=table_position,
                                   quat=actor.pose.quaternion)
            # Use table_name as geom name to avoid conflicts when multiple tables exist
            table.add_geom(name=f"{table_name}_geom",
                           type=mujoco.mjtGeom.mjGEOM_BOX, 
                           size=table_size,
                           condim=6, 
                           friction=[2, 0.04, 0.0005], # 0.8
                           priority=10,)
        else:
            raise TypeError(f"Unsupported primitive type: {type(actor)}")

    def _build_camera(self, cname:str, camera: CameraEntity):
        q_isaac_mujoco = t3d.quaternions.mat2quat(np.array([
            [ 0,  0, -1],
            [-1,  0,  0],
            [ 0,  1,  0]
        ]))
        W, H = camera.resolution
        fovy = 2 * np.arctan(H / (2 * camera.fy)) * 180 / np.pi
        if camera.mount == "eye_on_base":
            # cam_pose = self.task.layout.actors["robot"].pose * camera.pose
            cam_pose = camera.pose
            cam_q = t3d.quaternions.qmult(cam_pose.quaternion, q_isaac_mujoco)

            self.mj_worldbody.add_camera(
                name=cname,
                pos=cam_pose.position,
                quat=cam_q,
                fovy=fovy,
            )
        elif camera.mount == "eye_in_hand":
            self.mj_worldbody.add_camera(
                name=cname,
                pos=[1.5, 0., 0.8],  #  FIXME
                xyaxes=[0,1,0,-0.5,0,1], 
                fovy=fovy
            )
        elif camera.mount == "native":
            # The robot's own MJCF already has a <camera name=cname> element, properly attached
            # and posed by its asset authors (e.g. WidowX AI's wrist camera "cam" in
            # wxai_follower.xml) -- don't create a duplicate (MuJoCo requires unique camera
            # names), just leave it as-is. self.renderers[cname] still gets built from
            # sensor_cfgs (see reset()), so render() picks the existing camera up by name.
            existing_names = {c.name for c in self.mj_worldbody.find_all('camera')}
            assert cname in existing_names, (
                f"mount='native' camera '{cname}' not found among the robot's own MJCF cameras "
                f"{sorted(existing_names)} -- the sensor_cfgs key must match the physical "
                f"<camera name=...> element name."
            )
        elif camera.mount == "eye_in_head":
            torso_body = None
            for body in self.mj_worldbody.find_all('body'):
                if body.name == "torso_link":
                    torso_body = body
                    break
            assert torso_body is not None
            """ 
            I know this numbers look crazy!
            I obtain the first coordinate using isaacsim (g1_29dof_wholebody_dex3.usd)
            and obtain the second coordinates using mujoco (g1_29dof_wholebody_dex3.xml)
            and then i add them up by LUCK and it works! 
            """
            DEFAULT_HEAD_CAM_POSITION = np.array([0.05366004+0.0039635, 0.01752999 + 0, 0.4738702 + -0.044], dtype=np.float32)
            DEFAULT_HEAD_CAM_ORIENTATION = np.array([0.91496, 0.0, 0.40355, 0.0], dtype=np.float32)
            q = np.asarray(camera.pose.quaternion, dtype=np.float32)
            is_identity_quat = np.allclose(q[1:], 0.0, atol=1e-6) and np.isclose(abs(float(q[0])), 1.0, atol=1e-6)
            assert is_identity_quat, f"Expected eye_in_head camera quaternion to be identity (wxyz)"

            torso_body.add_camera(
                name=cname,
                pos=DEFAULT_HEAD_CAM_POSITION + camera.pose.position,  # FIXME
                quat=t3d.quaternions.qmult(DEFAULT_HEAD_CAM_ORIENTATION, q_isaac_mujoco),
                fovy=fovy
            )
        else:
            raise ValueError(f"Unsupported camera mount: {camera.mount}")


    def get_robot_qpos(self) -> dict[str,float]:
        from simple.robots.protocols import Controllable
        if not isinstance(self.task.robot, Controllable):
            raise TypeError("The task robot is not a Robot instance.")
        return self.task.robot.get_robot_qpos()
    
    def get_actuators_action(self) -> dict[str,float]:
        from simple.robots.protocols import Controllable
        if not isinstance(self.task.robot, Controllable):
            raise TypeError("The task robot is not a Robot instance.")
        return self.task.robot.get_actuators_action()
    
    def apply_action(self, action_cmd) -> None: # target_qpos
        applied_action=self.task.robot.apply_action(action_cmd) # target_qpos[:self.task.robot.dof]
        self.last_action = applied_action
    
    # def apply_action_command(self, action_command):
    #     for act in action_command:
    #         self.actuators[act[0]].ctrl = act[1]
    #         self.last_action[act[0]] = act[1]

    def set_robot_qpos(self, qpos):
        joint_names = list(self.joints.keys())

        if len(qpos) > len(joint_names):
            qpos = qpos[:len(joint_names)]
        
        for joint_name, q in zip(joint_names, qpos):
            self.joints[joint_name].qpos = q
            self.joints[joint_name].qvel = 0
            self.joints[joint_name].qacc = 0

    def mj_body_name(self, objtype: str) -> str:
        """MuJoCo body name for a layout actor key (e.g. "target_0"),
        mirroring the duplicate-label disambiguation applied in
        _setup_scene/_build_object (self._dup_object_labels): bare asset
        label unless multiple live actors share it, in which case
        f"{label}_{objtype}". External callers that need to address a
        specific object's MuJoCo body/joint by name (e.g. replay scripts
        restoring recorded object poses) must go through this rather than
        assuming the bare asset label -- with several same-asset instances
        (e.g. multiple "bin_b04" totes, or the industrial sorting task's
        screw/screwdriver copies), the bare label is ambiguous/missing.
        """
        actor = self.task.layout.actors[objtype]
        label = actor.asset.uid
        if isinstance(actor.asset, SemanticAnnotated):
            label = actor.asset.label
        return f"{label}_{objtype}" if label in self._dup_object_labels else label

    def set_object_poses(self, obj_names, obj_positions, obj_orientations):
        for _, (name, p, q) in enumerate(zip(obj_names, obj_positions, obj_orientations)):
            self.mjData.joint(f"{name}_joint").qpos = np.concatenate([p, q], axis=0)
    
    def _robot_mask_geom_ids(self) -> set[int]:
        from simple.core.actor import ObjectActor
        from simple.assets.primitive import Primitive

        excluded_geom_names = {"ground"}
        excluded_body_names = {"world"}

        dup_object_labels = getattr(self, "_dup_object_labels", set())
        for actor_name, actor in self.task.layout.actors.items():
            if isinstance(actor, ObjectActor):
                excluded_body_names.add(actor.asset.uid)
                obj_label = actor.asset.label if isinstance(actor.asset, SemanticAnnotated) else actor.asset.uid
                if obj_label in dup_object_labels:
                    excluded_body_names.add(f"{obj_label}_{actor_name}")
                else:
                    excluded_body_names.add(obj_label)
            elif isinstance(actor, Primitive):
                excluded_body_names.add(actor_name)
                excluded_geom_names.add(f"{actor_name}_geom")

        robot_geom_ids: set[int] = set()
        for geom_id in range(self.mj_physics_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_physics_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in excluded_geom_names:
                continue
            body_id = int(self.mj_physics_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.mj_physics_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in excluded_body_names:
                continue
            robot_geom_ids.add(int(geom_id))
        return robot_geom_ids

    def _render_robot_mask(self, renderer, camera_name: str, color: np.ndarray, robot_geom_ids: set[int]) -> np.ndarray:
        original_rgba = self.mj_physics_model.geom_rgba[sorted(robot_geom_ids)].copy()
        try:
            self.mj_physics_model.geom_rgba[sorted(robot_geom_ids), 3] = 0.0
            renderer.update_scene(
                self.mj_physics_data,
                scene_option=self.render_option,
                camera=camera_name,
            )
            background = renderer.render()[..., :3].astype(np.uint8)
        finally:
            self.mj_physics_model.geom_rgba[sorted(robot_geom_ids)] = original_rgba
            renderer.update_scene(
                self.mj_physics_data,
                scene_option=self.render_option,
                camera=camera_name,
            )

        color_i16 = color.astype(np.int16)
        background_i16 = background.astype(np.int16)
        return np.any(np.abs(color_i16 - background_i16) > 2, axis=-1)

    def render(self, render_robot_mask: bool | str = False) -> Dict[str, np.ndarray]:
        image_observations = {}
        mask_camera_name = None
        if isinstance(render_robot_mask, str):
            mask_camera_name = render_robot_mask
        elif render_robot_mask:
            mask_camera_name = "front_stereo_left"

        robot_geom_ids = self._robot_mask_geom_ids() if mask_camera_name is not None else set()
        for mjCamera in self.mj_worldbody.find_all('camera'):
            # A robot's own MJCF can carry cameras that aren't declared in the task's
            # sensor_cfgs (e.g. Aloha's "teleoperator_pov", WidowX's "cam") -- self.renderers
            # only has entries for sensor_cfgs-declared cameras, so skip anything else instead
            # of KeyError'ing.
            renderer = self.renderers.get(mjCamera.name)
            if renderer is None:
                continue
            
            # with self._telemetry.timer(f"render.updatescene.{mjCamera.name}"):
            renderer.update_scene(
                self.mjData, 
                scene_option=self.render_option, 
                camera=mjCamera.name
            )
            # with self._telemetry.timer(f"render.render.{mjCamera.name}"):
            render_product = renderer.render()
            color = render_product[..., :3].astype(np.uint8) if render_product.dtype != np.uint8 else render_product[..., :3]
            image_observations[mjCamera.name] = color

            # with self._telemetry.timer(f"render.mask.{mjCamera.name}"):
            if render_robot_mask and mjCamera.name == "front_stereo_left": # FIXME:
                renderer.enable_segmentation_rendering()
                try:
                    out = renderer.render()[...,0]
                    panda_geom_ids = []
                    for geom_id in np.unique(out):
                        if "panda" in self.mjModel.id2name(geom_id, 'geom'): # FIXME
                            panda_geom_ids.append(geom_id)

                    panda_mask = np.zeros_like(out, dtype=bool)
                    for pgid in panda_geom_ids:
                        panda_mask = np.logical_or(panda_mask, out == pgid)
                except:
                    robot_mask = np.ones_like(color[...,0], dtype=bool) # not sure why render seg fails
                image_observations["robot_mask"] = robot_mask
                
        return image_observations
    
    def close(self):
        # print("Closing Mujoco simulator...")
        if hasattr(self, "renderers") and self.renderers:
            for renderer in self.renderers.values():
                renderer.close()
            self.renderers = {}
