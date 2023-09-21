from gym.envs.registration import register

def register_envs():
    register(
        id='FishStationary',
        entry_point='cs285.envs.fish:FishStationary',
        max_episode_steps=768,
    )

    register(
        id='FishMoving',
        entry_point='cs285.envs.fish:FishMoving',
        max_episode_steps=768,
    )