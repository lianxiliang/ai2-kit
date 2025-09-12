from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashSteps, BashTemplate, make_gpu_parallel_steps
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import dict_nested_get, expand_globs, dump_json, list_split, flatten, create_fn, get_yaml
from ai2_kit.core.pydantic import BaseModel
# from ai2_kit.tool.dpdata import set_fparam, register_data_types

from typing import List, Tuple, Optional
from dataclasses import dataclass
from itertools import groupby
import os
import sys
import copy
import random
import dpdata
import numpy as np

from .iface import ICllTrainOutput, BaseCllContext, TRAINING_MODE
from .data import DataFormat, get_data_format, write_mace_cumulative_dataset
from .deepmd import make_deepmd_dataset

from .constant import (
    MACE_INPUT_FILE,
    MACE_FINAL_MODEL, 
    MACE_LMP_MODEL,
)

logger = get_logger(__name__)


class CllMaceInputConfig(BaseModel):

    model_num: int = 4
    """
    Total number of models to train.
    """
    init_dataset: List[str] = []
    """
    Dataset used to initialize training.
    """
    input_template: dict = dict()
    """
    MACE input template.
    """
    compress_model: bool = False
    """
    Whether to compress model after training.
    """
    pretrained_model: Optional[str] = None
    """
    Pretrained model used to finetune.
    """
    isolate_outliers: bool = False
    """
    If isolate_outliers is enabled, then outlier data will be separated from training data.
    """
    outlier_f_cutoff: float = 10.
    """
    The threshold of force magnitude to determine whether a data is outlier.
    """
    outlier_weight: float = 0.003
    """
    The weight of outlier data in training data.
    """

    fixture_models: List[str] = []
    """
    Fixture models used to initialize training, support glob pattern.
    If this is not empty, then the whole training process will be skipped.
    This feature is useful for debugging, or explore more structures without training.
    The models should be on the remote executor.
    The name fixture is used as the concept of fixture in pytest.
    """

    group_by_formula: bool = False
    """
    Grouping dataset by formula
    If this is enabled, then the dataset will be grouped by formula.
    Otherwise, the dataset will be grouped by ancestor.

    Set this to True when you have multiple structures with the same ancestor.
    """

    init_from_previous: bool = False
    """
    Use the previous models to initialize the current training,
    which can speed up the training process.
    """

    input_modifier_fn: Optional[str] = None
    """
    A python function to modify the input data.
    The function should take a input dict as input and return a dict.

    def input_modifier_fn(input: dict) -> dict:
        ...
        return new_input
    """

    ignore_error: bool = False
    """
    Ignore non critical errors.
    """

    mace_train_opts: str = ''
    """
    Extra options for mace train command
    """


class CllMaceContextConfig(BaseModel):
    script_template: BashTemplate
    mace_cmd: str = 'mace_run_train'
    concurrency: int = 5
    multi_gpus_per_job: bool = False


@dataclass
class CllMaceInput:
    config: CllMaceInputConfig
    mode: TRAINING_MODE
    type_map: List[str]
    sel_type: Optional[List[int]]
    old_dataset: List[Artifact]  # training data used by previous iteration
    new_dataset: List[Artifact]  # training data used by current iteration
    previous: List[Artifact]  # previous models used by previous iteration


@dataclass
class CllMaceContext(BaseCllContext):
    config: CllMaceContextConfig


@dataclass
class GenericMaceOutput(ICllTrainOutput):
    models: List[Artifact]
    dataset: List[Artifact]

    def get_mlp_models(self) -> List[Artifact]:
        return self.models

    def get_training_dataset(self) -> List[Artifact]:
        return self.dataset


async def cll_mace(input: CllMaceInput, ctx: CllMaceContext):
    executor = ctx.resource_manager.default_executor

    # if fix_models is set, return it and skip training
    if len(input.config.fixture_models) > 0:
        logger.info(f'Using fixture models: {input.config.fixture_models}')
        model_paths = executor.run_python_fn(expand_globs)(input.config.fixture_models)
        assert len(model_paths) > 0, f'No fixture models found: {input.config.fixture_models}'
        return GenericMaceOutput(
            dataset=input.old_dataset.copy(),
            models=[Artifact.of(url=url, format=DataFormat.MACE_MODEL)
                    for url in model_paths]
        )

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [new_dataset_dir, tasks_dir, outlier_dir] = executor.setup_workspace(
        work_dir, ['new_dataset', 'tasks', 'outlier_dataset'])

    init_dataset = ctx.resource_manager.resolve_artifacts(input.config.init_dataset)
    # input dataset contains data that generated by previous iteration
    # it should not contain the initial dataset
    input_dataset: List[Artifact] = input.old_dataset.copy()

    # make new dataset from raw data
    new_dataset, outlier_dataset = executor.run_python_fn(make_deepmd_dataset)(
        dataset_dir=new_dataset_dir,
        outlier_dir=outlier_dir,
        raw_data_collection=[a.to_dict() for a in input.new_dataset],
        type_map=input.type_map,
        isolate_outliers=input.config.isolate_outliers,
        outlier_f_cutoff=input.config.outlier_f_cutoff,
        group_by_formula=input.config.group_by_formula,
        deepmd_input_template={},   # not necessary for mace since we use simple MACE
        sel_type=input.sel_type,
        mode=input.mode,
        ignore_error=input.config.ignore_error,
    )

    input_dataset += [ Artifact.of(**a) for a in new_dataset]
    # use attrs to distinguish outlier dataset
    input_dataset += [ Artifact.of(**{**a, 'attrs': {**a['attrs'], 'outlier': True}}) for a in outlier_dataset]
    # classify dataset
    train_systems, outlier_systems, validation_systems = _classify_dataset(input_dataset + init_dataset)

    # for MACE, the cumulative xyz file will be written after each iteration
    # train_systems is a list of strings correspond to the data path
    all_datasets = input_dataset + init_dataset
    train_artifacts = [a for a in all_datasets if a.url in train_systems]

    # genertate cumulative dataset in the deepmd dataset dir, we basically keep the original deepmd dataset
    cumulative_train_file = executor.run_python_fn(make_mace_cumulative_dataset)(
        dataset_dir=new_dataset_dir, 
        dataset_collection=[a.to_dict() for a in train_artifacts], 
        type_map=input.type_map, 
        extxyzkey = ['ref_energy', 'ref_forces'],
    )

    # make task dirs
    mace_task_dirs = executor.run_python_fn(make_mace_task_dirs)(
        input_template=input.config.input_template,
        model_num=input.config.model_num,
        type_map=input.type_map,
        isolate_outliers=input.config.isolate_outliers,
        base_dir=tasks_dir,
        cumulative_train_file=cumulative_train_file,
        input_modifier_fn=input.config.input_modifier_fn,
    )


    all_steps: List[BashSteps] = []

    # run dp training jobs

    for i, task_dir in enumerate(mace_task_dirs):

        # FIXME: a temporary workaround to support previous model
        # Here we assume that the model are on the same cluster
        # it will fail if we support multiple cluster in the future
        previous_model = None
        if input.config.init_from_previous and input.previous:
            j = i % len(input.previous)
            previous_model = os.path.join(os.path.dirname(input.previous[j].url),
                                          MACE_FINAL_MODEL)

        steps = _build_mace_steps(
            mace_cmd=ctx.config.mace_cmd,
            compress_model=input.config.compress_model,
            cwd=task_dir,
            pretrained_model=input.config.pretrained_model,
            previous_model=previous_model,
            mace_train_opts=input.config.mace_train_opts,
            model_name='mace_model',  # Ensure consistent model name
        )
        all_steps.append(steps)

    # submit jobs by the number of concurrency
    jobs = []

    for i, steps_group in enumerate(list_split(all_steps, ctx.config.concurrency)):
        if not steps_group:
            continue

        if ctx.config.multi_gpus_per_job:
            script = BashScript(
                template=ctx.config.script_template,
                steps=make_gpu_parallel_steps(steps_group),
            )
        else:
            script = BashScript(
                template=ctx.config.script_template,
                steps=flatten(steps_group),
            )

        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)

    await gather_jobs(jobs, max_tries=2)

    logger.info(f'All models are trained, output dirs: {mace_task_dirs}')

    # collect trained models (both .model and -mliap_lammps.pt versions)
    models = []
    for i, task_dir in enumerate(mace_task_dirs):
        
        # MACE creates multiple model files:
        # 1. {name}.model - main MACE model
        # 2. {name}.model-mliap_lammps.pt - LAMMPS-optimized model (created by mace_create_lammps_model)
        
        # Choose appropriate model file based on compression setting
        if input.config.compress_model:
            final_model_file = os.path.join(task_dir, MACE_LMP_MODEL)
        else:
            final_model_file = os.path.join(task_dir, MACE_FINAL_MODEL)
        
        models.append(Artifact.of(
            url=final_model_file,
            format=DataFormat.MACE_MODEL,
            attrs={
                'logs_dir': os.path.join(task_dir, 'logs'),
                'checkpoints_dir': os.path.join(task_dir, 'checkpoints'),
            }
        ))

    return GenericMaceOutput(
        dataset=input_dataset.copy(),
        models=models
    )

def _classify_dataset(dataset: List[Artifact]):
    """
    Classify dataset into train, outlier and validation
    """
    train_systems: List[str] = []
    outlier_systems: List[str] = []
    validation_systems: List[str] = []

    for a in dataset:
        if dict_nested_get(a.attrs, ['deepmd', 'validation_data'], False):
            validation_systems.append(a.url)
        else:
            if dict_nested_get(a.attrs, ['outlier'], False):
                outlier_systems.append(a.url)
            else:
                train_systems.append(a.url)
    # FIXME: there are users reporting duplicated data in dataset
    # So we need to check and remove duplicated data
    # TODO: find the root cause of duplicated data and remove this workaround
    def _unique(l: list):
        r = sorted(set(l))
        if len(r) != len(l):
            logger.warning(f'Found duplicated data in dataset: {l}')
        return r
    return _unique(train_systems), _unique(outlier_systems), _unique(validation_systems)


def _build_mace_steps(mace_cmd: str,
                        compress_model: bool,
                        cwd: str,
                        previous_model: Optional[str] = None,
                        pretrained_model: Optional[str] = None,
                        mace_train_opts: str = '',
                        model_name: str = 'mace_model',
                        ):
    steps = []
    mace_train_cmd = f'{mace_cmd} --config {MACE_INPUT_FILE} {mace_train_opts}'

    # MACE restart logic: check for checkpoint files and restart accordingly
    if previous_model:
        mace_train_cmd = f'{mace_train_cmd} --restart-training {previous_model}'
    if pretrained_model:
        mace_train_cmd = f'{mace_train_cmd} --foundation-model {pretrained_model}'
    
    # MACE checkpoint restart logic
    mace_train_cmd_restart = f'if [ ! -f {model_name}.model ]; then {mace_train_cmd}; else echo "Model already trained, skipping..."; fi'

    steps.append(
        BashStep(cmd=mace_train_cmd_restart, cwd=cwd, checkpoint='mace-train')  # type: ignore
    )

    if compress_model:
        steps.append(BashStep(cmd=[f'mace_create_lammps_model', f'{model_name}.model', '--format=mliap'],
                              cwd=cwd))
    return steps


def make_mace_task_dirs(input_template: dict,
                          model_num: int,
                          type_map: List[str],
                          isolate_outliers: bool,
                          base_dir: str,
                          cumulative_train_file: str,
                          input_modifier_fn: Optional[str],
                          ):

    input_modifier = create_fn(input_modifier_fn, 'input_modifier_fn') if input_modifier_fn else lambda x: x

    mace_task_dirs = [os.path.join(base_dir, f'{i:03d}')  for i in range(model_num)]
    for task_dir in mace_task_dirs:
        os.makedirs(task_dir, exist_ok=True)
        mace_input = make_mace_input(
            input_template=input_template,
            type_map=type_map,
            isolate_outliers=isolate_outliers,
            train_file=cumulative_train_file,
        )

        mace_input = input_modifier(mace_input)
        
        # Convert config dict to YAML string format
        # Using StringIO because ruamel.yaml.dump() requires a file-like object
        yaml = get_yaml()
        from io import StringIO
        output = StringIO()
        yaml.dump(mace_input, output)
        mace_input_yaml = output.getvalue()
        
        mace_input_path = os.path.join(task_dir, MACE_INPUT_FILE)
        with open(mace_input_path, 'w') as f:
            f.write(mace_input_yaml)

    return mace_task_dirs


def make_mace_input(input_template: dict,
                      type_map: List[str],
                      isolate_outliers: bool,
                      train_file: str,
                      ):
    # create dp train input file
    """
    generate the input file, dynamic variables needs modification
    """

    def _random_seed():
        return random.randrange(sys.maxsize) % (1 << 32)

    mace_input = copy.deepcopy(input_template)

    mace_input['seed'] = _random_seed()
    mace_input['train_file'] = train_file

    mace_input['name'] = f"mace_model"
    
    return mace_input


def make_mace_cumulative_dataset(
        dataset_dir: str, 
        dataset_collection: List[ArtifactDict],
        type_map: List[str],
        extxyzkey: List[str] = ['ref_energy', 'ref_forces'],
):
    """create mace cumulative dataset since this is mace specific"""
    os.makedirs(dataset_dir, exist_ok=True)
    train_file = os.path.join(dataset_dir, 'train.xyz')

    # define a function to write the cumulative file in data.py
    write_mace_cumulative_dataset(train_file, dataset_collection, type_map, extxyzkey)
    logger.info(f'Created cumulative dataset: {train_file}')
    return train_file
