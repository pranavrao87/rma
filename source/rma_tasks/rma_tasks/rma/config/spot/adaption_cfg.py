from flat_env_cfg import SpotFlatEnvCfg



@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PrivCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0,1.0)
        )
        foot_force = ObsTerm(
            func=rma_mdp.contact_sensor,
            params = {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
        ground_friction = ObsTerm(
            func=rma_mdp.contact_friction,
            params = {"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SpotObsHistoryCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
    
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 12 # according to paper


    # observation groups
    priv_obs: PrivCfg=PrivCfg()
    policy: PolicyCfg = PolicyCfg()
    history: SpotObsHistoryCfg = SpotObsHistoryCfg()



@configclass
class SpotAdaptionCfg(SpotFlatEnvCfg):

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 10), origin_type="world", env_index=1000, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        #self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #    # terrain
    #     self.scene.terrain = TerrainImporterCfg(
    #         prim_path="/World/ground",
    #         terrain_type="generator",
    #         terrain_generator=rma_mdp.COBBLESTONE_ROAD_CFG,
    #         max_init_terrain_level=rma_mdp.COBBLESTONE_ROAD_CFG.num_rows - 1,
    #         collision_group=-1,
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             friction_combine_mode="multiply",
    #             restitution_combine_mode="multiply",
    #             static_friction=1.0,
    #             dynamic_friction=1.0,
    #         ),
    #         visual_material=sim_utils.MdlFileCfg(
    #             mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #             project_uvw=True,
    #             texture_scale=(0.25, 0.25),
    #         ),
    #         debug_vis=True,
    #     )

    #     # no height scan
    #     self.scene.height_scanner = None


class SpotFlatEnvCfg_PLAY(SpotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
 
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event

