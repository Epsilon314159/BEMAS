REGISTRY = {}


from .episode_runner import EpisodeRunner
from .curiosity_episode_runner import CuriosityEpisodeRunner


REGISTRY["episode"] = EpisodeRunner
REGISTRY["curiosity_episode"] = CuriosityEpisodeRunner


from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
