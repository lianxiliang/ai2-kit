from ai2_kit.core.artifact import ArtifactDict, Artifact

from typing import List, Tuple, Optional
from ase import Atoms

import ase.io
import os


class DataFormat:
    # customize data format
    CP2K_OUTPUT_DIR = 'cp2k/output_dir'
    VASP_OUTPUT_DIR = 'vasp/output_dir'
    LAMMPS_OUTPUT_DIR = 'lammps/output_dir'
    DEEPMD_OUTPUT_DIR = 'deepmd/output_dir'
    MACE_OUTPUT_DIR = 'mace/output_dir'
    ANYWARE_OUTPUT_DIR = 'anyware/output_dir'

    DEEPMD_MODEL = 'deepmd/model'
    MACE_MODEL = 'mace/model'
    DEEPMD_NPY = 'deepmd/npy'
    LASP_LAMMPS_OUT_DIR ='lasp+lammps/output_dir'

    # data format of dpdata
    CP2K_OUTPUT = 'cp2k/output'
    VASP_XML = 'vasp/xml'

    # data format of ase
    EXTXYZ = 'extxyz'
    VASP_POSCAR = 'vasp/poscar'


def get_data_format(artifact: dict) -> Optional[str]:
    """
    Get (or guess) data type from artifact dict
    Note: The reason of using dict instead of Artifact is Artifact is not pickleable
    """
    url = artifact.get('url')
    assert isinstance(url, str), f'url must be str, got {type(url)}'

    file_name = os.path.basename(url)
    format = artifact.get('format')
    if format and isinstance(format, str):
        return format  # TODO: validate format
    if file_name.endswith('.xyz'):
        return DataFormat.EXTXYZ
    if 'POSCAR' in file_name:
        return DataFormat.VASP_POSCAR
    return None


def artifacts_to_ase_atoms(artifacts: List[ArtifactDict], type_map: List[str]) -> List[Tuple[ArtifactDict, Atoms]]:
    """
    Read ase atoms list from artifacts
    Deprecated since it is not recommended to use ArtifactDict
    """
    results = []
    for a in artifacts:
        data_format = get_data_format(a)  # type: ignore
        url = a['url']
        if data_format in [DataFormat.VASP_POSCAR, 'vasp']:
            atoms_list = ase.io.read(url, ':', format='vasp')
        elif data_format in [DataFormat.EXTXYZ, 'extxyz']:
            atoms_list = ase.io.read(url, ':', format='extxyz')
        elif data_format is not None:
            atoms_list = ase.io.read(url, ':', format=data_format)
        else:
            raise ValueError(f'unsupported data format: {data_format}')
        results.extend((a, atoms) for atoms in atoms_list)
    return results


def artifacts_to_ase_atoms_v2(artifacts: List[Artifact]) -> List[Tuple[Artifact, Atoms]]:
    results = []
    for a in artifacts:
        data_format = get_data_format(a.to_dict())  # type: ignore
        url = a.url
        if data_format in [DataFormat.VASP_POSCAR, 'vasp']:
            atoms_list = ase.io.read(url, ':', format='vasp')
        elif data_format in [DataFormat.EXTXYZ, 'extxyz']:
            atoms_list = ase.io.read(url, ':', format='extxyz')
        elif data_format is not None:
            atoms_list = ase.io.read(url, ':', format=data_format)
        else:
            raise ValueError(f'unsupported data format: {data_format}')
        results.extend((a, atoms) for atoms in atoms_list)
    return results


def ase_atoms_to_cp2k_input_data(atoms: Atoms) -> Tuple[List[str], List[List[float]]]:
    coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms] # type: ignore
    cell = [list(row) for row in atoms.cell]  # type: ignore
    return (coords, cell)


def convert_to_lammps_input_data(systems: List[ArtifactDict], base_dir: str, type_map: List[str]):
    data_files = []
    atoms_list = artifacts_to_ase_atoms(systems, type_map=type_map)
    for i, (artifact, atoms) in enumerate(atoms_list):
        data_file = os.path.join(base_dir, f'{i:06d}.lammps.data')
        ase.io.write(data_file, atoms, format='lammps-data', specorder=type_map)  # type: ignore
        data_files.append({
            'url': data_file,
            'attrs': artifact['attrs'],
        })
    return data_files


def write_mace_cumulative_dataset(
        train_file: str,
        dataset_collection: List[ArtifactDict],
        type_map: List[str],
        extxyzkey: List[str] = ['energy', 'forces'],
        max_structures: Optional[int] = None,
        sample_method: str = 'sequential',
        shuffle_seed: Optional[int] = None
):
    """
    Write multiple datasets from artifact collection to a MACE-compatible extxyz file.
    
    Reads DeePMD format data from provided artifacts and converts them to ASE Atoms objects
    with energy and forces information, then writes them to a single extxyz file.
    
    :param train_file: Output path for the extxyz file
    :param dataset_collection: List of artifact dictionaries pointing to deepmd/npy format datasets
    :param type_map: List of chemical symbols mapping atom types 
    :param extxyzkey: List containing keys for energy and forces in the output file,
                      defaults to ['energy', 'forces']
    :param max_structures: Maximum number of structures to write, None means all
    :param sample_method: Method for sampling structures if max_structures is set:
                         'sequential' (default), 'random', 'even'
    :param shuffle_seed: Optional seed for shuffling the dataset order. Different seeds create
                         different training orders, which can increase ensemble diversity.
                         If None, no shuffling is performed.
    """
    from ai2_kit.tool.dpdata import read, deepmd2ase
    from ai2_kit.core.log import get_logger
    from ase import Atoms, io
    import random
    
    logger = get_logger(__name__)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(train_file)), exist_ok=True)
    
    datapaths = [a['url'] for a in dataset_collection]
    all_atoms = []
    
    # Process all dataset paths at once
    try:
        # Read all deepmd datasets
        systems = read(*datapaths)
        
        # Convert each system to ASE atoms with energy and forces
        for sys in systems:
            atoms_list = deepmd2ase(sys, energy_key=extxyzkey[0], forces_key=extxyzkey[1])
            all_atoms.extend(atoms_list)
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
    
    # Shuffle dataset if seed is provided (for ensemble diversity)
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(all_atoms)
        logger.info(f"Shuffled {len(all_atoms)} structures with seed {shuffle_seed}")
    
    # Sample structures if requested
    if max_structures is not None and max_structures < len(all_atoms):
        if sample_method == 'random':
            all_atoms = random.sample(all_atoms, max_structures)
        elif sample_method == 'even':
            # Take evenly spaced samples
            indices = [int(i * len(all_atoms) / max_structures) for i in range(max_structures)]
            all_atoms = [all_atoms[i] for i in indices]
        else:  # 'sequential'
            all_atoms = all_atoms[:max_structures]
        
        logger.info(f"Sampled {max_structures} structures using {sample_method} method")
    
    # Write all atoms to extxyz file
    if all_atoms:
        try:
            io.write(train_file, all_atoms, format='extxyz')
            logger.info(f"Wrote {len(all_atoms)} structures to {train_file}")
        except Exception as e:
            logger.error(f"Error writing to {train_file}: {e}")
    else:
        logger.warning("No structures were processed.")
    
    return train_file


def mace_xyz_to_dpdata(xyzfile: str, type_map: List[str]):
    """
    convert the mace xyz file for deepmd training, used for foundation model 
    
    :param xyzfile: xyz file path obtained from previous iter
    :type xyzfile: str
    :param type_map: type map
    :type type_map: List[str]
    """
    from ai2_kit.core.log import get_logger
    logger = get_logger(__name__)

    from ase.calculators.singlepoint import SinglePointCalculator
    import dpdata
    if not os.path.exists(xyzfile):
        raise FileNotFoundError(f'MACE output file not found: {xyzfile}')
    
    try: 
        atoms_list = ase.io.read(xyzfile, index=':')
    except Exception as e:
        raise ValueError(f'Failed to read XYZ file {xyzfile}: {e}')
    
    # Process frames and attach MACE results
    valid_atoms = []
    for i, atoms in enumerate(atoms_list):
        ener = atoms.info.get('MACE_energy')
        frc = atoms.arrays.get('MACE_forces')
        
        if ener is None or frc is None:
            logger.warning(f'Frame {i} missing MACE_energy or MACE_forces. Skipping.')
            continue
        
        atoms.calc = SinglePointCalculator(atoms, energy=ener, forces=frc)
        valid_atoms.append(atoms)

    if not valid_atoms:
        raise ValueError(f"No valid structures found in {xyzfile}")
    
    # Convert all valid frames to dpdata using dpdata's native ase/structure support
    labeled_systems = [dpdata.LabeledSystem(atoms, fmt='ase/structure') for atoms in valid_atoms]
    
    # Merge using dpdata's built-in method
    merged = labeled_systems[0]
    for ls in labeled_systems[1:]:
        merged += ls
    
    return merged    


