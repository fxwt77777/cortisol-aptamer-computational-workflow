# Computational workflow

The computational workflow is organized into two main branches: machine-learning-based candidate generation/ranking and molecular-dynamics-based structural evaluation.

## A. Machine-learning and sequence-ranking branch

1. Aggregate and audit Boltz-2-derived sequence-level outputs.
2. Define the Top/Bottom 5% extreme-label dataset.
3. Evaluate sequence similarity and prepare similarity-aware train/validation/test partitions.
4. Train Logistic Regression and Random Forest baselines.
5. Train and evaluate the CNN-BiLSTM model across multiple random seeds.
6. Generate candidate DNA aptamer sequences using the original evolutionary workflow.
7. Re-evaluate generated candidates with the downstream structural/affinity-ranking workflow.
8. Select representative candidates for MD analysis.

Relevant directories:

- `ml/boltz2_audit/`
- `ml/similarity_split/`
- `ml/baselines/`
- `ml/cnn_bilstm/`
- `ml/original_generation/`
- `ml/ga_rescore/`

## B. Molecular-dynamics branch

1. Prepare aptamer-cortisol starting structures.
2. Generate independent MD replicates with independent velocity seeds.
3. Perform NVT equilibration.
4. Perform NPT equilibration.
5. Perform 50-ns production MD.
6. Preprocess DNA-ligand trajectories.
7. Calculate DNA RMSD, radius of gyration, minimum DNA-ligand distance, contact counts, hydrogen bonds, ligand RMSD after DNA alignment, and DNA RMSF.
8. Calculate residue-level DNA-ligand contact fingerprints.
9. Perform replicate-level MM/GBSA calculations.
10. Summarize results across independent replicates.

Relevant directories:

- `md/setup/`
- `md/configs/`
- `md/analysis/`
- `md/mmgbsa/`
- `results_summary/md/`

## Independent-replicate design

For the newly generated 50-ns simulations, the recorded velocity seeds were:

| System | Replicate | Seed |
|---|---:|---:|
| aptC | Rep2 | 20260822 |
| aptC | Rep3 | 20260823 |
| APT3 | Rep2 | 20260832 |
| APT3 | Rep3 | 20260833 |
| APT4 | Rep2 | 20260842 |
| APT4 | Rep3 | 20260843 |

Each newly generated replicate used 0.5 ns NVT equilibration, 0.5 ns NPT equilibration, and 50 ns production MD.

## Statistical note

Independent MD trajectories are treated as independent replicates. Frames sampled from a single trajectory are temporally correlated and are not treated as independent replicates when estimating replicate-level uncertainty.

For MM/GBSA reporting, replicate-level means should be summarized across the independent trajectories, rather than treating individual trajectory frames as independent observations.
