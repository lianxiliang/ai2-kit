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


def _select_best_mace_model(base_dir: str, model_name: str = "mace_model") -> tuple[str, str]:
    """Select best MACE models for LAMMPS (prefers .pt) and deviation (requires .model)"""
    
    def find_first_existing(candidates, fallback):
        for filename in candidates:
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                return path
        return os.path.join(base_dir, fallback)
    
    # LAMMPS: prefer compressed .pt files
    lammps_model = find_first_existing([
        f"{model_name}_stagetwo.model-mliap_lammps.pt",
        f"{model_name}.model-mliap_lammps.pt", 
        f"{model_name}_stagetwo.model",
        f"{model_name}.model"
    ], f"{model_name}.model")
    
    # Deviation: only .model files (CLI requirement)
    deviation_model = find_first_existing([
        f"{model_name}_stagetwo.model",
        f"{model_name}.model"
    ], f"{model_name}.model")
    
    return lammps_model, deviation_model


def _get_mace_models_variables(models: List[Artifact]):
    """Generate template variables for MACE models with intelligent file selection"""
    vars = {}
    if not models:
        return vars
    
    lammps_models = []
    deviation_models = []
    
    for m in models:
        model_path = m.url
        model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        
        if model_dir:
            lammps_model, deviation_model = _select_best_mace_model(model_dir)
            lammps_models.append(lammps_model)
            deviation_models.append(deviation_model)
        else:
            # Use file as-is if no directory
            lammps_models.append(model_path)
            deviation_models.append(model_path)
    
    if lammps_models:
        vars['MACE_LAMMPS_MODEL'] = lammps_models[0]
        vars['MACE_MODELS'] = ' '.join(lammps_models)
    
    if deviation_models:
        vars['MACE_MODELS_FOR_DEVIATION'] = ' '.join(deviation_models)
        for i, m in enumerate(deviation_models):
            vars[f'MACE_MODELS_{i}'] = m
    
    return vars


async def cll_mace_lammps(input: CllMaceLammpsInput, ctx: CllMaceLammpsContext):
    """
    MACE-enabled LAMMPS exploration with integrated model deviation calculation.
    
    This function creates custom bash steps that run both LAMMPS simulation AND 
    MACE model deviation calculation in the SAME job submissions.
    """
    logger.info(f'Starting MACE-LAMMPS exploration with {len(input.mace_models)} models on device: {input.device}')
    
    # Log the model files being used
    for i, model in enumerate(input.mace_models):
        logger.info(f'MACE model {i+1}: {model.url}')
    
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
    
    # Extract SLURM environment prefix for MACE model deviation
    slurm_prefix = ""
    if 'lmp' in lammps_cmd_full:
        import re
        lmp_match = re.search(r'\blmp\s+', lammps_cmd_full) or re.search(r'\blmp(?=\s|$|[^a-zA-Z0-9_-])', lammps_cmd_full)
        if lmp_match and lmp_match.start() > 0:
            slurm_prefix = lammps_cmd_full[:lmp_match.start()].strip()
    
    # Build MACE model deviation command
    # Important: models_for_deviation is space-separated, must NOT be quoted to expand as multiple args
    models_for_deviation = mace_template_vars.get('MACE_MODELS_FOR_DEVIATION', '')
    type_map_str = ','.join(input.type_map) if input.type_map else ''
    
    # Build MACE model deviation command (CLI only)
    mace_cmd_parts = ['mace-model-devi', '--models', models_for_deviation]
    mace_cmd_parts.extend(['--traj', 'traj.lammpstrj', '--output', 'model_devi.out', '--device', input.device])
    
    if type_map_str:
        mace_cmd_parts.extend(['--type-map', f'"{type_map_str}"'])
    
    base_mace_cmd = ' '.join(mace_cmd_parts)
    mace_cmd = f'{slurm_prefix} {base_mace_cmd}' if slurm_prefix else base_mace_cmd
    
    # Create combined bash steps (LAMMPS + MACE model deviation in same job)
    base_lammps_cmd = f'{ctx.config.lammps_cmd} -i lammps.input'
    
    steps = []
    for task_dir in task_dirs:
        # Multi-line bash script for better readability
        script_lines = [
            '# Run LAMMPS simulation',
            f'if [ -f md.restart.* ]; then',
            f'    {base_lammps_cmd} -v restart 1',
            f'else',
            f'    {base_lammps_cmd} -v restart 0',
            f'fi',
            '',
            '# Calculate MACE model deviation',
            f'{mace_cmd}',
        ]
        
        combined_cmd = '\n'.join(script_lines)
        
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
