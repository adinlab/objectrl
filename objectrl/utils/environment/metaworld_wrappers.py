import gymnasium as gym


class SparsifyRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Meta-World environments
    that replaces reward with the binary success signal.

    Args:
        env : gym.Env
            The Meta-World environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Replace reward with success (-1.0 or 0.0)
        reward = info["success"] - 1.0

        return obs, reward, terminated, truncated, info
