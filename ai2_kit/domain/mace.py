from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashSteps, BashTemplate, make_gpu_parallel_steps
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.log import get_logger
from ai2_kit.core.util import dict_nested_get, expand_globs, dump_json, list_split, flatten, create_fn
from ai2_kit.core.pydantic import BaseModel
from ai2_kit.tool.dpdata import set_fparam, register_data_types

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
from .data import DataFormat, get_data_format

from .constant import (
    DP_CHECKPOINT_FILE,
    DP_DISP_FILE,
    DP_PROFILING_FILE,
    DP_INPUT_FILE,
    DP_FROZEN_MODEL,
    DP_ORIGINAL_MODEL,
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
    Deepmd input template.
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

    dp_train_opts: str = ''
    """
    Extra options for dp train command
    """


class CllMaceContextConfig(BaseModel):
    script_template: BashTemplate
    dp_cmd: str = 'mace_run_train'
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
    

    # make task dirs
    dp_task_dirs, dw_task_dir = executor.run_python_fn(make_deepmd_task_dirs)(
        input_template=input.config.input_template,
        model_num=input.config.model_num,
        type_map=input.type_map,
        train_systems=train_systems,
        outlier_systems=outlier_systems,
        validation_systems=validation_systems,
        outlier_weight=input.config.outlier_weight,
        isolate_outliers=input.config.isolate_outliers,
        dw_input_template=dw_input_template,
        base_dir=tasks_dir,
        input_modifier_fn=input.config.input_modifier_fn,
    )


    all_steps: List[BashSteps] = []
    # run dp training jobs

    for i, task_dir in enumerate(dp_task_dirs):
        executor.mkdir(task_dir)
        # FIXME: a temporary workaround to support previous model
        # Here we assume that the model are on the same cluster
        # it will fail if we support multiple cluster in the future
        previous_model = None
        if input.config.init_from_previous and input.previous:
            j = i % len(input.previous)
            previous_model = os.path.join(os.path.dirname(input.previous[j].url),
                                          DP_ORIGINAL_MODEL)

        steps = _build_deepmd_steps(
            dp_cmd=ctx.config.dp_cmd,
            compress_model=input.config.compress_model,
            cwd=task_dir,
            pretrained_model=input.config.pretrained_model,
            previous_model=previous_model,
            dp_train_opts=input.config.dp_train_opts,
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

    logger.info(f'All models are trained, output dirs: {dp_task_dirs}')
    return GenericDeepmdOutput(
        dataset=input_dataset.copy(),
        models=[Artifact.of(
            url=os.path.join(url, DP_FROZEN_MODEL),
            format=DataFormat.DEEPMD_MODEL,
        ) for url in dp_task_dirs]
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


def _build_deepmd_steps(dp_cmd: str,
                        compress_model: bool,
                        cwd: str,
                        previous_model: Optional[str] = None,
                        pretrained_model: Optional[str] = None,
                        dp_train_opts: str = '',
                        ):
    steps = []
    dp_train_cmd = f'{dp_cmd} train {dp_train_opts} {DP_INPUT_FILE}'
    if previous_model:
        dp_train_cmd = f'{dp_train_cmd} -f {previous_model}'
    if pretrained_model:
        dp_train_cmd = f'{dp_train_cmd} --finetune {pretrained_model}'

    dp_train_cmd_restart = f'if [ ! -f model.ckpt.index ]; then {dp_train_cmd}; else {dp_cmd} train {DP_INPUT_FILE} --restart model.ckpt; fi'

    steps.append(
        BashStep(cmd=dp_train_cmd_restart, cwd=cwd, checkpoint='dp-train')  # type: ignore
    )
    if compress_model:
        steps.append(BashStep(cmd=[dp_cmd, 'freeze', '-o', DP_ORIGINAL_MODEL, '&&',
                                   dp_cmd, 'compress', '-i', DP_ORIGINAL_MODEL, '-o', DP_FROZEN_MODEL],
                              cwd=cwd))
    else:
        # FIXME: a temporary workaround to support previous model
        steps.append(BashStep(cmd=[dp_cmd, 'freeze', '-o', DP_ORIGINAL_MODEL, '&&',
                                   'cp', DP_ORIGINAL_MODEL, DP_FROZEN_MODEL],
                              cwd=cwd))
    return steps


def make_deepmd_task_dirs(input_template: dict,
                          model_num: int,
                          type_map: List[str],
                          train_systems: List[str],
                          outlier_systems: List[str],
                          validation_systems: List[str],
                          isolate_outliers: bool,
                          outlier_weight: float,
                          dw_input_template: Optional[dict],
                          base_dir: str,
                          input_modifier_fn: Optional[str],
                          ):

    input_modifier = create_fn(input_modifier_fn, 'input_modifier_fn') if input_modifier_fn else lambda x: x

    dp_task_dirs = [os.path.join(base_dir, f'{i:03d}')  for i in range(model_num)]
    for task_dir in dp_task_dirs:
        os.makedirs(task_dir, exist_ok=True)
        dp_input = make_deepmd_input(
            input_template=input_template,
            type_map=type_map,
            train_systems=train_systems,
            outlier_systems=outlier_systems,
            validation_systems=validation_systems,
            isolate_outliers=isolate_outliers,
            outlier_weight=outlier_weight,
        )

        # If dw_model is set in dp_input,
        # it will create a softlink named dw_model.pb to the current dir,
        # and then modify the value in dw_model to dw_model.pb.
        dw_model = dict_nested_get(dp_input, ['model', 'modifier', 'model_name'], None)
        if dw_model is not None:
            # Create a soft link named 'dw_model.pb' to the file specified by dw_model
            link_target = os.path.join(task_dir, 'dw_model.pb')
            os.system(f'ln -sf {dw_model} {link_target}')
            # Modify the value in dw_model to 'dw_model.pb'
            dp_input['model']['modifier']['model_name'] = 'dw_model.pb'

        dp_input = input_modifier(dp_input)
        dp_input_path = os.path.join(task_dir, DP_INPUT_FILE)
        dump_json(dp_input, dp_input_path)

    dw_task_dir = None
    if dw_input_template:
        dw_task_dir = os.path.join(base_dir, 'train_dw')
        os.makedirs(dw_task_dir, exist_ok=True)
        dw_input = make_deepmd_input(
            input_template=dw_input_template,
            type_map=type_map,
            train_systems=train_systems,
            outlier_systems=outlier_systems,
            validation_systems=validation_systems,
            isolate_outliers=isolate_outliers,
            outlier_weight=outlier_weight,
        )
        dw_input_path = os.path.join(dw_task_dir, DP_INPUT_FILE)
        dump_json(dw_input, dw_input_path)

    return dp_task_dirs, dw_task_dir


def make_deepmd_input(input_template: dict,
                      type_map: List[str],
                      train_systems: List[str],
                      outlier_systems: List[str],
                      validation_systems: List[str],
                      isolate_outliers: bool,
                      outlier_weight: float,
                      ):
    # create dp train input file
    # ref: https://github.com/deepmodeling/dpgen2/blob/master/examples/ch4/param_CH4_deepmd-kit-2.1.1.json
    # ref: https://github.com/deepmodeling/dpgen2/blob/master/dpgen2/op/prep_dp_train.py
    # ref: https://github.com/deepmodeling/dpgen2/blob/master/dpgen2/op/run_dp_train.py

    def _random_seed():
        return random.randrange(sys.maxsize) % (1 << 32)

    dp_input = copy.deepcopy(input_template)
    training: dict = dp_input['training']

    # set output files
    training['disp_file'] = DP_DISP_FILE
    training['save_ckpt'] = DP_CHECKPOINT_FILE
    training['profiling_file'] = DP_PROFILING_FILE

    # set random seed
    discriptor = dp_input['model']['descriptor']
    if discriptor['type'] == 'hybrid':
        for d in discriptor['list']:
            d['seed'] = _random_seed()
    else:
        discriptor['seed'] = _random_seed()
    dp_input['model']['fitting_net']['seed'] = _random_seed()
    dp_input['training']['seed'] = _random_seed()

    auto_prob_str = "prob_sys_size"
    if isolate_outliers and len(outlier_systems) > 0:
        # if isolate_outlier is enabled, then we need to set different weight for outlier data
        # e.g: prob_sys_size;0:57:0.9997;57:60:0.0003
        auto_prob_str += f';0:{len(train_systems)}:{1-outlier_weight}'
        auto_prob_str += f';{len(train_systems)}:{len(train_systems)+len(outlier_systems)}:{outlier_weight}'

    # v2 training data
    training_data = training.get('training_data', {})
    training_data = {
        'set_prefix': 'set',
        'batch_size': 'auto',
        'sys_probs': None,
        ** training_data,
        'systems': train_systems + outlier_systems,
        'auto_prob_style': auto_prob_str,
    }
    training['training_data'] = training_data

    # ignore validation section if no data is provided, or else dp will throw error
    # OSError: [Errno cannot find valid a data system] Please check your setting for data systems
    if len(validation_systems) > 0:
        validation_data = {
            'systems': validation_systems,
            'set_prefix': training_data['set_prefix'],
            'batch_size': training_data['batch_size'],
        }
        training['validation_data'] = validation_data

    # other params
    dp_input['model']['type_map'] = type_map

    return dp_input

