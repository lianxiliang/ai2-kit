from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger

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
    cll_lammps,
)
from ai2_kit.tool.mace_devi import calculate_mace_model_deviation

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
    preset_template: str
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
    MACE-enabled LAMMPS exploration using custom_ff approach
    """
    logger.info(f'Starting MACE-LAMMPS exploration with {len(input.mace_models)} models on device: {input.device}')
    
    # Generate MACE template variables (same pattern as DeepMD)
    mace_template_vars = _get_mace_models_variables(input.mace_models)
    
    # Prepare LAMMPS configuration with MACE template variables
    config_dict = input.config.dict()
    config_dict['template_vars'] = {
        **input.config.template_vars,
        **mace_template_vars,
    }
    
    lammps_config = CllLammpsInputConfig(**config_dict)
    
    # Create LAMMPS input using MACE preset template
    lammps_input = CllLammpsInput(
        config=lammps_config,
        type_map=input.type_map,
        mass_map=input.mass_map,
        mode=input.mode,
        preset_template=input.preset_template,
        new_system_files=input.new_system_files or [],
        dp_models={},  # Empty - we use MACE through preset template
        dp_modifier=None,
        dp_sel_type=None,
    )
    
    # Create LAMMPS context
    lammps_ctx = CllLammpsContext(
        config=ctx.config,
        path_prefix=ctx.path_prefix,
        resource_manager=ctx.resource_manager,
    )
    
    # Run the existing LAMMPS workflow (no modifications needed!)
    logger.info('Running LAMMPS simulation with MACE force field')
    lammps_output = await cll_lammps(lammps_input, lammps_ctx)
    
    # Post-process: Calculate MACE model deviation only
    logger.info('Post-processing: calculating MACE model deviation')
    mace_outputs = []
    executor = ctx.resource_manager.default_executor
    mace_models_paths = [m.url for m in input.mace_models]
    
    for lammps_artifact in lammps_output.model_devi_outputs:
        task_dir = lammps_artifact.url
        
        # Standard MACE-LAMMPS simulation (no FEP complexity)
        traj_path = os.path.join(task_dir, 'traj.lammpstrj')
        model_devi_path = os.path.join(task_dir, 'model_devi.out')
        
        # Calculate MACE model deviation
        if os.path.exists(traj_path):
            try:
                executor.run_python_fn(calculate_mace_model_deviation)(
                    model_files=mace_models_paths,
                    traj_file=traj_path,
                    output_file=model_devi_path,
                    type_map=input.type_map,
                    device=input.device,  # Pass device for MACE calculator
                )
                logger.debug(f'MACE model deviation calculated on device {input.device}: {model_devi_path}')
            except Exception as e:
                logger.warning(f'Failed to calculate MACE model deviation for {traj_path}: {e}')
                if not input.config.ignore_error:
                    raise
        
        # Create artifact with MACE-specific attributes
        mace_artifact = Artifact.of(
            url=task_dir,
            format=DataFormat.LAMMPS_OUTPUT_DIR,
            attrs={
                **lammps_artifact.attrs,
                'model_devi_file': 'model_devi.out',
                'structures': 'traj.lammpstrj',
                'mace_models_count': len(input.mace_models),
                'force_field': 'mace',
            }
        )
        mace_outputs.append(mace_artifact)
    
    logger.info(f'MACE-LAMMPS exploration completed. Generated {len(mace_outputs)} outputs')
    return GenericMaceLammpsOutput(model_devi_outputs=mace_outputs)
