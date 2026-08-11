# Machine-Learning-Guided Generation of DNA Aptamer Candidates with Enhanced Computational Binding to Cortisol

This repository contains custom code and reproducibility materials associated with the manuscript.

## About

This repository contains the research prototype accompanying our manuscript. The implementation is under active development and has not yet undergone peer review.


## Repository structure

- `ml/baselines/`: Logistic Regression and Random Forest baselines.
- `ml/cnn_bilstm/`: CNN-BiLSTM training, evaluation, correlation analysis, and paired bootstrap comparison.
- `ml/original_generation/`: original sequence-generation workflow.
- `ml/similarity_split/`: Top/Bottom 5% preparation and similarity-aware splitting.
- `ml/boltz2_audit/`: audit of Boltz-2-derived affinity-related metrics.
- `ml/ga_rescore/`: generated-candidate rescoring and comparison.
- `md/setup/`: independent MD replicate preparation and seed metadata.
- `md/configs/`: GROMACS NVT, NPT, and 50-ns production parameter files.
- `md/analysis/`: trajectory preprocessing and structural/contact analyses.
- `md/mmgbsa/`: replicate-level MM/GBSA processing.
- `results_summary/md/`: compact derived MD summary tables.
- `docs/`: workflow, reproducibility, and availability documentation.

## Python environment

Create the environment with:

```bash
conda env create -f environment.yml
conda activate cortisol-aptamer-code
```

Alternatively:

```bash
pip install -r requirements.txt
```

## External software

Parts of the workflow require third-party software that is not redistributed here:

- GROMACS
- MMseqs2
- CD-HIT
- gmx_MMPBSA
- Boltz-2-related software or services

Independent-replicate MD calculations used GROMACS 2022.3 (conda-forge build).

## Credentials

No passwords, SSH credentials, API keys, or machine-specific private paths are stored in this repository. Scripts requiring an NVIDIA-hosted endpoint read the credential from the environment variable `NVIDIA_API_KEY`.

## Large data

Large MD trajectories, model checkpoints, and intermediate simulation files are intentionally excluded from Git because individual files can be several gigabytes. The repository retains the custom code, simulation parameters, run metadata, and compact derived results needed to document the computational workflow.

## Reproducibility

Random seeds and run configurations are retained where stochastic procedures were used. Independent MD trajectories are analyzed separately, and uncertainty across MD replicates should be estimated from replicate-level results rather than by treating frames from one trajectory as independent replicates.

## Code and data availability

See:

- `docs/CODE_AVAILABILITY.txt`
- `docs/DATA_AVAILABILITY.md`
- `docs/WORKFLOW.md`
- `docs/MANUSCRIPT_CODE_MAP.md`
- `docs/REPRODUCIBILITY.md`

## Citation

Size Tong, Jin Wang.  
*Machine-Learning-Guided Generation of DNA Aptamer Candidates with Enhanced Computational Binding to Cortisol.*

Citation information will be added after the manuscript receives final bibliographic information.

