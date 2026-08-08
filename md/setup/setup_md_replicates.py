from pathlib import Path
import os
import csv
import re
import shutil

ROOT = Path(
    os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
    "md_replicates_3x50ns_20260806"
)

if ROOT.exists():
    raise FileExistsError(
        f"Target directory already exists: {ROOT}"
    )

systems = {
    "aptC": {
        "source": Path(
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_test_aptC_10ns"
        ),
        "production_mdp": "mdp/md_10ns.mdp",
        "rep1": (
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_final_aptC60_apt3apt4_50_20260526/"
            "aptC_ionmix_60ns"
        ),
        "seeds": {
            "rep2": 20260822,
            "rep3": 20260823,
        },
    },
    "APT3": {
        "source": Path(
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_boundstart_apt3apt4_50_20260524_144252/"
            "apt3_ionmix_boundstart_50ns/gmx"
        ),
        "production_mdp": "mdp/md_ionmix_50ns.mdp",
        "rep1": (
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_boundstart_apt3apt4_50_20260524_144252/"
            "apt3_ionmix_boundstart_50ns/gmx"
        ),
        "seeds": {
            "rep2": 20260832,
            "rep3": 20260833,
        },
    },
    "APT4": {
        "source": Path(
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_boundstart_apt3apt4_50_20260524_144252/"
            "apt4_ionmix_boundstart_50ns/gmx"
        ),
        "production_mdp": "mdp/md_ionmix_50ns.mdp",
        "rep1": (
            os.environ.get("APTAMER_PROJECT_ROOT", "./data") + "/"
            "ionmix_boundstart_apt3apt4_50_20260524_144252/"
            "apt4_ionmix_boundstart_50ns/gmx"
        ),
        "seeds": {
            "rep2": 20260842,
            "rep3": 20260843,
        },
    },
}


def replace_parameter(text, parameter, value):
    pattern = re.compile(
        rf"(?mi)^(\s*{re.escape(parameter)}\s*=\s*).*$"
    )

    if not pattern.search(text):
        raise ValueError(
            f"Parameter not found in MDP: {parameter}"
        )

    return pattern.sub(
        rf"\g<1>{value}",
        text,
        count=1,
    )


manifest = []

for system_name, config in systems.items():
    source = config["source"]

    required = [
        source / "em_ionmix.gro",
        source / "system.top",
        source / "index_clean.ndx",
        source / "mdp" / "nvt.mdp",
        source / "mdp" / "npt.mdp",
        source / config["production_mdp"],
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing source files:\n"
            + "\n".join(missing)
        )

    for replicate, seed in config["seeds"].items():
        destination = ROOT / system_name / replicate
        mdp_dir = destination / "mdp"
        logs_dir = destination / "logs"

        mdp_dir.mkdir(parents=True)
        logs_dir.mkdir()

        shutil.copy2(
            source / "em_ionmix.gro",
            destination / "em_ionmix.gro",
        )

        shutil.copy2(
            source / "system.top",
            destination / "system.top",
        )

        shutil.copy2(
            source / "index_clean.ndx",
            destination / "index_clean.ndx",
        )

        nvt_text = (
            source / "mdp" / "nvt.mdp"
        ).read_text(encoding="utf-8")

        nvt_text = replace_parameter(
            nvt_text,
            "gen_seed",
            seed,
        )

        nvt_text = replace_parameter(
            nvt_text,
            "gen_vel",
            "yes",
        )

        nvt_text = replace_parameter(
            nvt_text,
            "continuation",
            "no",
        )

        (mdp_dir / "nvt.mdp").write_text(
            nvt_text,
            encoding="utf-8",
        )

        shutil.copy2(
            source / "mdp" / "npt.mdp",
            mdp_dir / "npt.mdp",
        )

        production_text = (
            source / config["production_mdp"]
        ).read_text(encoding="utf-8")

        # aptC原文件为10 ns；统一改成50 ns：
        # 25,000,000 steps × 0.002 ps = 50,000 ps。
        production_text = replace_parameter(
            production_text,
            "nsteps",
            25000000,
        )

        production_text = replace_parameter(
            production_text,
            "continuation",
            "yes",
        )

        production_text = replace_parameter(
            production_text,
            "gen_vel",
            "no",
        )

        (mdp_dir / "md_50ns.mdp").write_text(
            production_text,
            encoding="utf-8",
        )

        manifest.append({
            "system": system_name,
            "replicate": replicate,
            "velocity_seed": seed,
            "starting_structure": (
                str(source / "em_ionmix.gro")
            ),
            "rep1_existing_directory": config["rep1"],
            "new_directory": str(destination),
            "production_nsteps": 25000000,
            "dt_ps": 0.002,
            "production_time_ns": 50,
            "gromacs_version": "2022.3-conda_forge",
        })


ROOT.mkdir(parents=True, exist_ok=True)

manifest_path = ROOT / "replicate_manifest.csv"

with manifest_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=manifest[0].keys(),
    )
    writer.writeheader()
    writer.writerows(manifest)

print("[OK] Created:", ROOT)
print("[OK] Manifest:", manifest_path)

for row in manifest:
    print(
        row["system"],
        row["replicate"],
        "seed=",
        row["velocity_seed"],
    )
