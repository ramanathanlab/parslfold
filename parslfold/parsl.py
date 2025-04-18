"""Utilities to build Parsl configurations."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal
from typing import Sequence
from typing import Union

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider
from parsl.providers import PBSProProvider
from pydantic import Field

from parslfold.utils import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore [assignment]


class BaseComputeConfig(BaseModel, ABC):
    """Compute config (HPC platform, number of GPUs, etc)."""

    # Name of the platform to uniquely identify it
    name: Literal[''] = ''

    @abstractmethod
    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class WorkstationConfig(BaseComputeConfig):
    """Compute config for a GPU workstation."""

    name: Literal['workstation'] = 'workstation'  # type: ignore[assignment]

    available_accelerators: int | Sequence[str] = Field(
        default=1,
        description='Number of GPU accelerators to use.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for workstation execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address=address_by_hostname(),
                    label='htex',
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=(10000, 20000),
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class PolarisConfig(BaseComputeConfig):
    """Polaris@ALCF configuration.

    See here for details: https://docs.alcf.anl.gov/polaris/workflows/parsl/
    """

    name: Literal['polaris'] = 'polaris'  # type: ignore[assignment]

    num_nodes: int = Field(
        default=1,
        description='Number of nodes to request.',
    )
    worker_init: str = Field(
        default='',
        description='Command to be run before starting a worker. '
        'Load any modules and environments, etc.',
    )
    scheduler_options: str = Field(
        default='#PBS -l filesystems=home:eagle:grand',
        description='PBS directives, pass -J for array jobs.',
    )
    account: str = Field(
        ...,
        description='The account to charge compute to.',
    )
    queue: str = Field(
        ...,
        description='Which queue to submit jobs to, will usually be prod.',
    )
    walltime: str = Field(
        ...,
        description='Maximum job time.',
    )
    cpus_per_node: int = Field(
        default=32,
        description='Up to 64 with multithreading.',
    )
    cores_per_worker: float = Field(
        default=8,
        description='Number of cores per worker. '
        'Evenly distributed between GPUs.',
    )
    retries: int = Field(
        default=0,
        description='Number of retries upon failure.',
    )
    worker_debug: bool = Field(
        default=False,
        description='Enable worker debug.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a parsl configuration for running on Polaris@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: PathLike
            Directory in which to store Parsl run files.
        """
        return Config(
            executors=[
                HighThroughputExecutor(
                    label='htex',
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=self.worker_debug,
                    # available_accelerators will override settings
                    # for max_workers
                    available_accelerators=4,
                    cores_per_worker=self.cores_per_worker,
                    # address=address_by_interface('bond0'),
                    cpu_affinity='block-reverse',
                    prefetch_capacity=0,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind',
                            overrides='--depth=64 --ppn 1',
                        ),
                        account=self.account,
                        queue=self.queue,
                        select_options='ngpus=4',
                        # PBS directives: for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        # Command to be run before starting a worker, such as:
                        worker_init=self.worker_init,
                        # number of compute nodes allocated for each block
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Increase to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            run_dir=str(run_dir),
            # checkpoint_mode='task_exit',
            retries=self.retries,
            app_cache=True,
        )


class AuroraConfig(BaseComputeConfig):
    """Aurora@ALCF configuration.

    See here for details: https://docs.alcf.anl.gov/aurora/workflows/parsl/
    """

    name: Literal['aurora'] = 'aurora'  # type: ignore[assignment]

    num_nodes: int = Field(
        default=1,
        description='Number of nodes to request.',
    )
    worker_init: str = Field(
        default='',
        description='Command to be run before starting a worker. '
        'Load any modules and environments, etc.',
    )
    scheduler_options: str = Field(
        default='#PBS -l filesystems=home:flare',
        description='PBS directives, pass -J for array jobs.',
    )
    account: str = Field(
        ...,
        description='The account to charge compute to.',
    )
    queue: str = Field(
        ...,
        description='Which queue to submit jobs to, will usually be prod.',
    )
    walltime: str = Field(
        ...,
        description='Maximum job time.',
    )
    cpus_per_node: int = Field(
        default=208,
        description='208 virtual CPUs per node with multithreading.',
    )
    cores_per_worker: float = Field(
        default=17,
        description='Number of cores per worker. '
        'Evenly distributed between GPUs, leaving 4 cores idle.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries upon failure.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a parsl configuration for running on Aurora@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: PathLike
            Directory in which to store Parsl run files.
        """
        tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

        return Config(
            executors=[
                HighThroughputExecutor(
                    # Ensures one worker per GPU tile on each node
                    available_accelerators=tile_names,
                    max_workers_per_node=12,  # number of GPUs on a node
                    cores_per_worker=self.cores_per_worker,
                    # Distributes threads to workers/tiles optimized for Aurora
                    cpu_affinity='list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204',
                    # Increase if you have many more tasks than workers
                    prefetch_capacity=0,
                    # Options that specify properties of PBS Jobs
                    provider=PBSProProvider(
                        # Project name
                        account=self.account,
                        # Submission queue
                        queue=self.queue,
                        # Commands run before workers launched
                        # Activate your environment where Parsl is installed
                        worker_init=self.worker_init,
                        # Wall time for batch jobs
                        walltime=self.walltime,
                        # Change if data/modules located on other filesystem
                        scheduler_options=self.scheduler_options,
                        # Ensures 1 manger per node;
                        # the manager will distribute work to its 12 workers,
                        # one per tile
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind',
                            overrides='--ppn 1',
                        ),
                        # options added to #PBS -l select aside from ncpus
                        select_options='',
                        # Number of nodes per PBS job
                        nodes_per_block=self.num_nodes,
                        # Min number of concurrent PBS jobs running workflow
                        min_blocks=0,
                        # Max number of concurrent PBS jobs running workflow
                        max_blocks=1,
                        # Hardware threads per node
                        cpus_per_node=self.cpus_per_node,
                    ),
                ),
            ],
            # How many times to retry failed tasks
            # this is necessary if you have tasks that are interrupted by a PBS
            # so that they will restart in the next job
            retries=self.retries,
            run_dir=str(run_dir),
            # Enable app cache for better performance on Aurora
            app_cache=True,
        )


ComputeConfigs = Union[
    WorkstationConfig,
    PolarisConfig,
    AuroraConfig,
]
