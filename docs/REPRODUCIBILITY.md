# Reproducibility notes

## Machine-learning analyses

The repository retains the custom scripts used for preprocessing, baseline modeling, CNN-BiLSTM training and evaluation, sequence generation, similarity analysis, and candidate rescoring.

Where stochastic procedures were used, random seeds and run configurations are retained in the corresponding scripts or metadata files.

The revised CNN-BiLSTM workflow uses repeated training across multiple random seeds. Baseline comparisons use the same predefined held-out test partition where applicable.

## Molecular-dynamics replicates

The newly generated MD replicates use independent velocity-generation seeds:

| System | Replicate | Seed |
|---|---:|---:|
| aptC | Rep2 | 20260822 |
| aptC | Rep3 | 20260823 |
| APT3 | Rep2 | 20260832 |
| APT3 | Rep3 | 20260833 |
| APT4 | Rep2 | 20260842 |
| APT4 | Rep3 | 20260843 |

For these newly generated replicates, the simulation stages are:

- NVT equilibration: 0.5 ns
- NPT equilibration: 0.5 ns
- Production MD: 50 ns

The corresponding GROMACS parameter files are stored under:

`md/configs/`

The replicate metadata are stored in:

`md/setup/replicate_manifest.csv`

## MD analysis

Each independent trajectory is analyzed separately.

The repository includes scripts for:

- trajectory preprocessing;
- DNA RMSD;
- radius of gyration;
- minimum DNA-ligand distance;
- DNA-ligand contact counts;
- DNA-ligand hydrogen bonds;
- ligand RMSD after DNA alignment;
- DNA residue-level RMSF;
- residue-level DNA-ligand contact analysis.

Compact derived replicate summaries are retained under:

`results_summary/md/`

## Replicate-level uncertainty

Frames sampled from one MD trajectory are temporally correlated and are not treated as indepent replicates.

For replicate-level MM/GBSA reporting, each independent trajectory should first yield its own replicate-level estimate. Summary uncertainty should then be calculated across the independent replicate-level estimates.

## External software

Reproduction of the complete workflow additionally requires third-party software described in `README.md`, including GROMACS, MMseqs2, CD-HIT, and gmx_MMPBSA where applicable.

Exact results can also depend on software versions, GPU/CUDA configuration, and external model or service versions.
