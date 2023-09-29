from d4rl_slim_benchmark.algorithms.offline.cql import make_trainer as make_cql_trainer
from d4rl_slim_benchmark.algorithms.offline.iql import make_trainer as make_iql_trainer

ALGORITHMS = {
    "IQL": make_iql_trainer,
    "CQL": make_cql_trainer,
}


def get_algo_factory(algo_name):
    return ALGORITHMS[algo_name.upper()]
