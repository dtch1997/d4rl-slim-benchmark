# type: ignore
"""lxm3 launch script for hpo"""
import importlib
import os
import sys

from absl import app
from absl import flags
from absl import logging
from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl
from ml_collections import config_flags

_LAUNCH_ON_CLUSTER = flags.DEFINE_bool(
    "launch_on_cluster", False, "If true, launch on cluster"
)
_ENTRYPOINT = flags.DEFINE_string("entrypoint", None, "Entrypoint to run")
_IMAGE = flags.DEFINE_string(
    "image", os.environ.get("LXM_SINGULARITY_IMAGE"), "Path to container image."
)
_EXP_NAME = flags.DEFINE_string("exp_name", None, "Name of experiment")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", "d4rl_slim_benchmark", "wandb project")
_WANDB_ENTITY = flags.DEFINE_string("wandb_entity", "dtch1997", "wandb user")
_WANDB_GROUP = flags.DEFINE_string("wandb_group", "{xid}_{name}", "wandb group")
_WANDB_MODE = flags.DEFINE_string("wandb_mode", "online", "wandb mode")
config_flags.DEFINE_config_file("config", None, "Path to config")
flags.mark_flags_as_required(["config"])

FLAGS = flags.FLAGS


def _get_wandb_env_vars(work_unit: xm.WorkUnit, experiment_name: str):
    xid = work_unit.experiment_id
    wid = work_unit.work_unit_id
    env_vars = {
        "WANDB_PROJECT": _WANDB_PROJECT.value,
        "WANDB_ENTITY": _WANDB_ENTITY.value,
        "WANDB_NAME": f"{experiment_name}_{xid}_{wid}",
        "WANDB_MODE": _WANDB_MODE.value,
    }
    if _WANDB_GROUP.value is not None:
        env_vars.update(
            {
                "WANDB_RUN_GROUP": _WANDB_GROUP.value.format(
                    name=experiment_name, xid=xid
                )
            }
        )
    try:
        import git

        commit_sha = git.Repo().head.commit.hexsha
        env_vars["WANDB_GIT_REMOTE_URL"] = git.Repo().remote().url
        env_vars["WANDB_GIT_COMMIT"] = commit_sha
    except Exception:
        logging.info("Unable to parse git info.")
    return env_vars


def _get_hyper():
    sweep_file = config_flags.get_config_filename(FLAGS["config"])
    sys.path.insert(0, os.path.abspath(os.path.dirname(sweep_file)))
    sweep_module, _ = os.path.splitext(os.path.basename(sweep_file))
    m = importlib.import_module(sweep_module)
    sys.path.pop(0)
    if hasattr(m, "get_sweep"):
        return m.get_sweep(None)
    else:
        return [{}]


def main(_):
    with xm_cluster.create_experiment(experiment_title="hpo") as experiment:
        exp_name = _EXP_NAME.value
        if exp_name is None:
            exp_name = _ENTRYPOINT.value.split(".")[-1]

        requirements = xm_cluster.JobRequirements(ram=16 * xm.GB, gpu=1)
        if not _LAUNCH_ON_CLUSTER.value:
            executor = xm_cluster.Local(requirements)
        else:
            executor = ucl.UclGridEngine(
                requirements=requirements,
                walltime=20 * xm.Hr,
            )

        env_vars = {
            "MUJOCO_GL": "osmesa",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(0.7),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }

        config_resource = xm_cluster.Fileset(
            files={config_flags.get_config_filename(FLAGS["config"]): "config.py"}
        )

        spec = xm_cluster.PythonPackage(
            entrypoint=xm_cluster.ModuleName(_ENTRYPOINT.value),
            path="..",
            resources=[config_resource],
        )
        spec = xm_cluster.SingularityContainer(spec, image_path=_IMAGE.value)

        args = {"config": config_resource.get_path("config.py", executor.Spec())}
        overrides = config_flags.get_override_values(FLAGS["config"])
        overrides = {f"config.{k}": v for k, v in overrides.items()}
        logging.info("Overrides: %r", overrides)
        args.update(overrides)

        [executable] = experiment.package(
            [xm.Packageable(spec, executor.Spec(), env_vars=env_vars, args=args)]
        )

        async def make_job(work_unit: xm.WorkUnit, **args):
            job = xm.Job(
                executable,
                executor,
                args=args,
                env_vars=_get_wandb_env_vars(work_unit, exp_name),
            )

            work_unit.add(job)

        with experiment.batch():
            for parameters in _get_hyper():
                experiment.add(make_job, parameters)


if __name__ == "__main__":
    app.run(main)
