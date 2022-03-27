from gym.envs.registration import register

register(
    id='STLPathPlan-v1',
    entry_point='gym_pathplan.envs:STL_Problem_Preprocess',
)