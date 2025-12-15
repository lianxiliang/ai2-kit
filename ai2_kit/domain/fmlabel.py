from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_sample, dump_json
from ai2_kit.core.log import get_logger
from ai2_kit.core.pydantic import BaseModel

from typing import List, Tuple, Literal
from dataclasses import dataclass
from ase import Atoms, io

import os

from .data import DataFormat, artifacts_to_ase_atoms
from .iface import ICllLabelOutput, BaseCllContext


logger = get_logger(__name__)


class CllFmLabelInputConfig(BaseModel):
    init_system_files: List[str] = []

    fm_model_path: str
    """path to foundation models"""

    limit: int = 50
    """
    Limit of the number of systems to be labeled.
    """

    limit_method: Literal["even", "random", "truncate"] = "even"
    
    device: Literal['cuda', 'cpu'] = 'cuda'

    default_dtype: str = 'float64'
    """Default dtype for MACE evaluation."""
    
    batch_size: int = 1
    """Batch size for MACE evaluation."""

    ignore_error: bool = False
    """
    Ignore error when running FM labeling.
    """

class CllFmLabelContextConfig(BaseModel):
    script_template: BashTemplate
    fm_cmd: str = 'mace_eval_configs'
    concurrency: int = 1


@dataclass
class CllFmLabelInput:
    config: CllFmLabelInputConfig
    system_files: List[Artifact]
    type_map: List[str]
    initiated: bool = False  # FIXME: this seems to be a bad design idea


@dataclass
class CllFmLabelContext(BaseCllContext):
    config: CllFmLabelContextConfig


@dataclass
class GenericFmLabelOutput(ICllLabelOutput):
    FmLabel_outputs: List[Artifact]

    def get_labeled_system_dataset(self):
        return self.FmLabel_outputs


async def cll_fmlabel(input: CllFmLabelInput, ctx: CllFmLabelContext) -> GenericFmLabelOutput:
    executor = ctx.resource_manager.default_executor

    # For the first round
    # FIXME: move out from this function, this should be done in the workflow
    if not input.initiated:
        input.system_files += ctx.resource_manager.resolve_artifacts(input.config.init_system_files)

    if len(input.system_files) == 0:
        return GenericFmLabelOutput(FmLabel_outputs=[])

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [tasks_dir] = executor.setup_workspace(work_dir, ['tasks'])

    # create task dirs and prepare input files
    fm_task_dir = executor.run_python_fn(make_fm_task_dir)(
        system_files=[a.to_dict() for a in input.system_files],
        type_map=input.type_map,
        base_dir=tasks_dir,
        # initialize all data if not initiated
        limit=0 if not input.initiated else input.config.limit,
        limit_method=input.config.limit_method,
    )

    # build the running command
    cmd = '\n'.join([
        f"{ctx.config.fm_cmd} \\",
        f"  --configs input.xyz \\",
        f"  --model {input.config.fm_model_path} \\",
        f"  --output output.xyz \\",
        f"  --device {input.config.device} \\",
        f"  --default_dtype {input.config.default_dtype} \\",
        f"  --batch_size {input.config.batch_size}",
    ])

    step = BashStep(
        cwd=fm_task_dir['url'],
        cmd=cmd,
        checkpoint='fmlabel',
        exit_on_error=not input.config.ignore_error,
    )


    # Submit job (single task, but use list for consistency)
    script = BashScript(
        template=ctx.config.script_template,
        steps=[step],
    )
    job = executor.submit(script.render(), cwd=tasks_dir)
    await gather_jobs([job], max_tries=2)

    # return xyz as output as MACE_OUTPUT_DIR for mat
    # data conversion will be handled by deepmd.py
    fm_outputs = [Artifact.of(
        url=fm_task_dir['url'],
        format=DataFormat.MACE_OUTPUT_DIR,
        executor=executor.name,
        attrs=fm_task_dir['attrs'],
    )]

    return GenericFmLabelOutput(FmLabel_outputs=fm_outputs)



def make_fm_task_dir(system_files: List[ArtifactDict],
                        type_map: List[str],
                        base_dir: str,
                        limit: int = 0,
                        limit_method: Literal["even", "random", "truncate"] = "even",
                        ) -> ArtifactDict:
    """create the task dir for doing mace evaluation: generate a combined XYZ file """
    atoms_list: List[Tuple[ArtifactDict, Atoms]] = artifacts_to_ase_atoms(system_files, type_map=type_map)

    if limit > 0:
        atoms_list = list_sample(atoms_list, limit, method=limit_method)

    # create tha task dir
    task_dir = os.path.join(base_dir, 'fmlabel')
    os.makedirs(task_dir, exist_ok=True)
    
    # Write combined extxyz file for MACE evaluation
    input_xyz = os.path.join(task_dir, 'input.xyz')
    allatoms = [atoms for _, atoms in atoms_list]
    io.write(input_xyz, allatoms, format='extxyz')
    logger.info(f"Created combined input.xyz with {len(allatoms)} structures in {task_dir}")

    # collect file information for traceback
    data_file = [debugdata for debugdata, _ in atoms_list]
    dump_json(data_file, os.path.join(task_dir, 'debug.data-file.json'))
    # Collect attributes from first input file
    combined_attrs = atoms_list[0][0]['attrs'] if atoms_list else {}

    return {
        'url': task_dir,
        'attrs': combined_attrs,
    }   # type: ignore
