from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.script import BashStep, BashScript, make_gpu_parallel_steps
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_split

from typing import List, Optional, Mapping
from dataclasses import dataclass
import os

from .iface import BaseCllContext, ICllExploreOutput, TRAINING_MODE
from .data import DataFormat
from .lammps import (
    CllLammpsInputConfig, 
    CllLammpsContextConfig,
    CllLammpsInput,
    CllLammpsContext,
    cll_lammps,  # Import back for reuse
)

logger = get_logger(__name__)


# MACE-LAMMPS uses the same configuration as regular LAMMPS
CllMaceLammpsInputConfig = CllLammpsInputConfig
CllMaceLammpsContextConfig = CllLammpsContextConfig


@dataclass
class CllMaceLammpsInput:
    """MACE-LAMMPS input data structure"""
    config: CllMaceLammpsInputConfig
    mace_models: List[Artifact]  # MACE committee models
    type_map: List[str]
    mass_map: List[float]
    mode: TRAINING_MODE = 'default'
    new_system_files: Optional[List[Artifact]] = None
    device: str = 'cuda'  # Device for MACE model deviation calculation
    @classmethod
    def from_mace_template(cls, config: CllMaceLammpsInputConfig, 
                          mace_models: List[Artifact], 
                          type_map: List[str], 
                          mass_map: List[float],
                          input_template: dict,
                          **kwargs) -> 'CllMaceLammpsInput':
        """
        Create CllMaceLammpsInput with device extracted from MACE input template
        
        Args:
            input_template: MACE configuration template containing device setting
            **kwargs: Other parameters
        """
        device = input_template.get('device', 'cuda')  # Extract device from template
        
        return cls(
            config=config,
            mace_models=mace_models,
            type_map=type_map,
            mass_map=mass_map,
            device=device,
            **kwargs
        )


@dataclass
class CllMaceLammpsContext(BaseCllContext):
    """MACE-LAMMPS execution context"""
    config: CllMaceLammpsContextConfig


@dataclass
class GenericMaceLammpsOutput(ICllExploreOutput):
    """MACE-LAMMPS output data structure"""
    model_devi_outputs: List[Artifact]

    def get_model_devi_dataset(self):
        return self.model_devi_outputs


def _get_mace_models_variables(models: List[Artifact]):
    """
    Generate template variables for MACE models following DeepMD pattern
    """
    vars = {}
    
    if models:
        # Use the first LAMMPS-optimized model for simulation
        first_model = models[0]
        
        # Get LAMMPS-compatible model path
        base_path = first_model.url
        if 'mliap_lammps.pt' in base_path:
            vars['MACE_LAMMPS_MODEL'] = base_path
        else:
            vars['MACE_LAMMPS_MODEL'] = base_path.replace('.model', '.model-mliap_lammps.pt')
        
        # Store all models for post-processing and model deviation
        model_paths = [m.url for m in models]
        vars['MACE_MODELS'] = ' '.join(model_paths)
        
        # Individual model variables for flexibility
        for i, m in enumerate(models):
            vars[f'MACE_MODELS_{i}'] = m.url
    
    return vars


async def cll_mace_lammps(input: CllMaceLammpsInput, ctx: CllMaceLammpsContext):
    """
    MACE-enabled LAMMPS exploration with integrated model deviation calculation.
    
    This function creates custom bash steps that run both LAMMPS simulation AND 
    MACE model deviation calculation in the SAME job submissions.
    """
    logger.info(f'Starting MACE-LAMMPS exploration with {len(input.mace_models)} models on device: {input.device}')
    
    # We need to create our own bash steps instead of calling cll_lammps
    # because we need to modify the commands BEFORE job submission
    
    executor = ctx.resource_manager.default_executor
    
    # Setup workspace (same as cll_lammps)
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    
    # Generate MACE template variables
    mace_template_vars = _get_mace_models_variables(input.mace_models)
    
    # Import required functions from lammps module
    from .lammps import make_lammps_task_dirs
    
    # Get data files: use new_system_files if available, otherwise fall back to system_files
    if input.new_system_files and len(input.new_system_files) > 0:
        data_files = input.new_system_files
    else:
        data_files = ctx.resource_manager.resolve_artifacts(input.config.system_files)
    
    assert len(data_files) > 0, 'no data files found for MACE-LAMMPS exploration'
    
    # Create LAMMPS task directories (reuse lammps logic)
    tasks_dir, task_dirs = executor.run_python_fn(make_lammps_task_dirs)(
        combination_vars=input.config.explore_vars,
        broadcast_vars=input.config.broadcast_vars,
        data_files=[a.to_dict() for a in data_files],
        dp_models={},  # Empty - we use MACE through preset template
        n_steps=input.config.nsteps,
        timestep=input.config.timestep,
        sample_freq=input.config.sample_freq,
        no_pbc=input.config.no_pbc,
        n_wise=input.config.n_wise,
        ensemble=input.config.ensemble,
        fix_statement=input.config.fix_statement,
        preset_template=input.config.preset_template or 'mace',
        input_template=input.config.input_template,
        plumed_config=input.config.plumed_config,
        extra_template_vars={**input.config.template_vars, **mace_template_vars},
        type_map=input.type_map,
        mass_map=input.mass_map,
        type_alias=input.config.type_alias,
        work_dir=work_dir,
        mode=input.mode,
        dp_modifier=None,
        dp_sel_type=None,
        fep_opts=input.config.fep_opts,
        custom_ff=input.config.custom_ff,
    )
    
    # Extract SLURM environment prefix from lammps_cmd for MACE model deviation
    lammps_cmd_full = ctx.config.lammps_cmd
    
    # Parse SLURM prefix (everything before the actual 'lmp' command)
    # Example: "srun --environment=mace-lmp-plumed --container-workdir=$PWD --cpu-bind=socket --ntasks=1 --gres=gpu:1 lmp ..."
    # We want: "srun --environment=mace-lmp-plumed --container-workdir=$PWD --cpu-bind=socket --ntasks=1 --gres=gpu:1"
    slurm_prefix = ""
    if 'lmp' in lammps_cmd_full:
        # Use regex to find 'lmp' followed by space or arguments (actual lmp command, not part of other words)
        import re
        lmp_match = re.search(r'\blmp\s+', lammps_cmd_full)
        if not lmp_match:
            # Try to find 'lmp' at end of string or followed by non-alphanumeric
            lmp_match = re.search(r'\blmp(?=\s|$|[^a-zA-Z0-9_-])', lammps_cmd_full)
        
        if lmp_match and lmp_match.start() > 0:
            # Extract everything before 'lmp' as the SLURM prefix
            slurm_prefix = lammps_cmd_full[:lmp_match.start()].strip()
            logger.info(f'Extracted SLURM prefix for MACE: {slurm_prefix}')
    
    # Prepare MACE model deviation command components  
    mace_models_str = ' '.join([m.url for m in input.mace_models])
    type_map_str = ','.join(input.type_map) if input.type_map else ''
    
    # Build MACE model deviation command using standalone package
    # Use mace-model-devi command directly (cleaner than ai2-kit tool)
    mace_cmd_args = [
        'mace-model-devi',
        '--models', f'"{mace_models_str}"',
        '--traj', 'traj.lammpstrj',
        '--output', 'model_devi.out',
        '--device', input.device,
    ]
    if type_map_str:
        mace_cmd_args.extend(['--type-map', f'"{type_map_str}"'])
    
    base_mace_cmd = ' '.join(mace_cmd_args)
    
    # Apply SLURM prefix to MACE command if needed
    if slurm_prefix:
        # When using SLURM container, use standalone mace-model-deviation package directly
        # No conda activation needed - package should be installed in container
        mace_cmd = f'{slurm_prefix} {base_mace_cmd}'
    else:
        # For non-SLURM environments, use standalone package directly
        mace_cmd = base_mace_cmd
    
    # Create combined bash steps (LAMMPS + MACE model deviation in same job)
    base_lammps_cmd = f'{ctx.config.lammps_cmd} -i lammps.input'
    
    steps = []
    for task_dir in task_dirs:
        # LAMMPS simulation command
        lammps_cmd = f'if [ -f md.restart.* ]; then {base_lammps_cmd} -v restart 1; else {base_lammps_cmd} -v restart 0; fi'
        
        # Combined command: LAMMPS followed by MACE model deviation
        combined_cmd = f'{lammps_cmd} && {mace_cmd}'
        
        # Single bash step that does both operations
        steps.append(BashStep(
            cwd=task_dir['url'],
            cmd=combined_cmd,
            checkpoint='mace-lammps-combined',
            exit_on_error=not input.config.ignore_error
        ))
    
    # Submit jobs with the combined steps (same as cll_lammps)
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
            
        if ctx.config.multi_gpus_per_job:
            script = BashScript(
                template=ctx.config.script_template,
                steps=make_gpu_parallel_steps(steps_group),  # type: ignore
            )
        else:
            script = BashScript(
                template=ctx.config.script_template,
                steps=steps_group,
            )
            
        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)
    
    logger.info(f'Submitted {len(jobs)} combined MACE-LAMMPS jobs')
    
    # Wait for ALL jobs to complete (both LAMMPS + model deviation)
    await gather_jobs(jobs, max_tries=2)
    
    # Build outputs - both LAMMPS and model deviation are complete
    outputs = []
    for task_dir in task_dirs:
        mace_artifact = Artifact.of(
            url=task_dir['url'],
            format=DataFormat.LAMMPS_OUTPUT_DIR,
            attrs={
                **task_dir['attrs'],
                'model_devi_file': 'model_devi.out',
                'structures': 'traj.lammpstrj',
                'mace_models_count': len(input.mace_models),
                'force_field': 'mace',
            }
        )
        outputs.append(mace_artifact)
    
    logger.info(f'MACE-LAMMPS exploration completed. Generated {len(outputs)} outputs with integrated model deviation')
    return GenericMaceLammpsOutput(model_devi_outputs=outputs)
