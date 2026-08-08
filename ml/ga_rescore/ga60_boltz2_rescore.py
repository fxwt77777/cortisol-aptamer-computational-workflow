#!/usr/bin/env python3
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import math
import random

import httpx
from fastapi import HTTPException

RUN_TAG = "run1"  # 这一套工程的名字/标签

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{task_id}"
PUBLIC_URL = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"

# 每条请求之间的固定间隔（秒）
PER_REQUEST_DELAY_SECONDS = 25


async def make_nvcf_call(
    function_url: str,
    data: Dict[str, Any],
    additional_headers: Optional[Dict[str, Any]] = None,
    NVCF_POLL_SECONDS: int = 300,
    MANUAL_TIMEOUT_SECONDS: int = 400,
) -> Dict:
    """
    调用 NVIDIA Cloud Functions（长轮询）
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 NVIDIA_API_KEY 为你的 API Key")

    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "NVCF-POLL-SECONDS": f"{NVCF_POLL_SECONDS}",
            "Content-Type": "application/json",
        }
        if additional_headers is not None:
            headers.update(additional_headers)

        logger.debug(
            f"Headers: { {k: v for k, v in headers.items() if 'Authorization' not in k} }"
        )
        logger.debug(f"Making NVCF call to {function_url}")
        logger.debug(f"Data: {data}")

        response = await client.post(
            function_url,
            json=data,
            headers=headers,
            timeout=MANUAL_TIMEOUT_SECONDS,
        )
        logger.debug(f"NVCF response: {response.status_code, response.headers}")

        if response.status_code == 202:
            task_id = response.headers.get("nvcf-reqid")
            if not task_id:
                raise HTTPException(500, "Missing nvcf-reqid in 202 response")

            while True:
                status_response = await client.get(
                    STATUS_URL.format(task_id=task_id),
                    headers=headers,
                    timeout=MANUAL_TIMEOUT_SECONDS,
                )
                if status_response.status_code == 200:
                    return status_response.status_code, status_response
                elif status_response.status_code in [400, 401, 404, 422, 500]:
                    raise HTTPException(
                        status_response.status_code,
                        "Error while waiting for function: ",
                        status_response.text,
                    )

        elif response.status_code == 200:
            return response.status_code, response
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)


# 随机生成 n 条长度在 [min_len, max_len] 的 DNA 序列
def generate_random_dna_sequences(
    n: int = 50, min_len: int = 50, max_len: int = 65
) -> List[str]:
    seqs: List[str] = []
    bases = "ACGT"
    for _ in range(n):
        L = random.randint(min_len, max_len)
        seq = "".join(random.choice(bases) for _ in range(L))
        seqs.append(seq)
    return seqs


# 从返回 JSON 提取亲和力结果
def get_affinity_values(response_dict: Dict[str, Any]):
    affinities = response_dict.get("affinities", {})
    ligand_aff = affinities.get("B", {})

    pred_list = ligand_aff.get("affinity_pred_value", [])
    prob_list = ligand_aff.get("affinity_probability_binary", [])
    pic50_list = ligand_aff.get("affinity_pic50", [])

    pred = float(pred_list[0]) if pred_list else None
    prob = float(prob_list[0]) if prob_list else None
    pic50 = float(pic50_list[0]) if pic50_list else None

    if pred is not None:
        ic50_uM = 10 ** pred          # IC50[µM]
        ic50_M = ic50_uM * 1e-6      # IC50[M]
    else:
        ic50_uM = None
        ic50_M = None

    return pred, ic50_uM, ic50_M, prob, pic50


async def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description=(
            "Rescore the 60 GA-generated DNA candidates "
            "using the original NVIDIA Boltz-2 HCY workflow."
        )
    )

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            project_root
            / "legacy_original_20251225"
            / "data"
            / "ga60_boltz2_rescore_input.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            script_dir
            / "ga60_boltz2_rescore_results.xlsx"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N pending sequences. "
            "Useful for a smoke test."
        ),
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    failure_path = (
        output_path.parent
        / "ga60_boltz2_rescore_failures.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    source_df = pd.read_csv(input_path)

    required_columns = {
        "candidate_id",
        "sequence",
        "length_nt",
        "pred_strong_prob_cls",
        "in_diverse10",
        "in_best5",
    }

    missing_columns = (
        required_columns - set(source_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing input columns: "
            + ", ".join(sorted(missing_columns))
        )

    source_df["candidate_id"] = (
        source_df["candidate_id"]
        .astype(str)
        .str.strip()
    )

    source_df["sequence"] = (
        source_df["sequence"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if source_df["candidate_id"].duplicated().any():
        raise ValueError(
            "Duplicate candidate_id values detected"
        )

    if source_df["sequence"].duplicated().any():
        raise ValueError(
            "Duplicate DNA sequences detected"
        )

    valid_dna = source_df["sequence"].str.fullmatch(
        r"[ACGT]+"
    )

    if not valid_dna.all():
        invalid = source_df.loc[
            ~valid_dna,
            ["candidate_id", "sequence"],
        ]

        raise ValueError(
            "Invalid DNA sequences detected:\n"
            + invalid.to_string(index=False)
        )

    calculated_lengths = (
        source_df["sequence"].str.len()
    )

    if not (
        calculated_lengths
        == source_df["length_nt"].astype(int)
    ).all():
        raise ValueError(
            "Reported and calculated sequence lengths differ"
        )

    if len(source_df) != 60:
        raise ValueError(
            f"Expected 60 GA candidates, found "
            f"{len(source_df)}"
        )

    if not (calculated_lengths == 65).all():
        raise ValueError(
            "Not all GA candidates are 65 nt"
        )

    if output_path.exists():
        completed_df = pd.read_excel(output_path)

        if "dna_sequence" not in completed_df.columns:
            raise ValueError(
                "Existing output does not contain "
                "dna_sequence column"
            )

        completed_df["dna_sequence"] = (
            completed_df["dna_sequence"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        completed_sequences = set(
            completed_df["dna_sequence"]
        )
    else:
        completed_df = pd.DataFrame()
        completed_sequences = set()

    pending_df = source_df.loc[
        ~source_df["sequence"].isin(
            completed_sequences
        )
    ].copy()

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "--limit must be at least 1"
            )

        pending_df = pending_df.head(
            args.limit
        )

    print(
        f"Input candidates: {len(source_df)}"
    )
    print(
        f"Already completed: "
        f"{len(completed_sequences)}"
    )
    print(
        f"Pending in this run: "
        f"{len(pending_df)}"
    )
    print(
        f"Output: {output_path}"
    )

    if pending_df.empty:
        print(
            "[OK] No pending sequences. "
            "Nothing to submit."
        )
        return

    source_order = {
        sequence: order
        for order, sequence in enumerate(
            source_df["sequence"]
        )
    }

    pending_records = pending_df.to_dict(
        orient="records"
    )

    successful_this_run = 0
    failed_this_run = 0

    for position, record in enumerate(
        pending_records,
        start=1,
    ):
        candidate_id = str(
            record["candidate_id"]
        )

        sequence = str(
            record["sequence"]
        )

        print()
        print(
            "=" * 80
        )
        print(
            f"{candidate_id}: "
            f"{position}/{len(pending_records)}"
        )
        print(
            f"Length: {len(sequence)} nt"
        )
        print(
            f"DNA: {sequence}"
        )

        data = {
            "polymers": [
                {
                    "id": "A",
                    "molecule_type": "dna",
                    "sequence": sequence,
                }
            ],
            "ligands": [
                {
                    "id": "B",
                    "ccd": "HCY",
                    "predict_affinity": True,
                }
            ],
            "recycling_steps": 3,
            "sampling_steps": 50,
            "diffusion_samples": 1,
            "step_scale": 1.638,
            "sampling_steps_affinity": 50,
            "diffusion_samples_affinity": 1,
            "output_format": "mmcif",
        }

        try:
            code, response = await make_nvcf_call(
                function_url=PUBLIC_URL,
                data=data,
            )

            if code != 200:
                raise RuntimeError(
                    f"HTTP status {code}: "
                    f"{response.text[:500]}"
                )

            response_dict = response.json()

            confidence_scores = (
                response_dict.get(
                    "confidence_scores",
                    [],
                )
            )

            confidence_score = (
                float(confidence_scores[0])
                if confidence_scores
                else None
            )

            (
                affinity_prediction,
                ic50_uM,
                ic50_M,
                affinity_likelihood,
                affinity_pic50,
            ) = get_affinity_values(
                response_dict
            )

            result_record = {
                "candidate_id": candidate_id,
                "dna_sequence": sequence,
                "length_nt": len(sequence),
                "ga_score": record.get(
                    "score"
                ),
                "pred_strong_prob_cls": (
                    record.get(
                        "pred_strong_prob_cls"
                    )
                ),
                "GC_content": record.get(
                    "GC_content"
                ),
                "max_run": record.get(
                    "max_run"
                ),
                "in_diverse10": record.get(
                    "in_diverse10"
                ),
                "in_best5": record.get(
                    "in_best5"
                ),
                "confidence_score": (
                    confidence_score
                ),
                (
                    "affinity_pred_value_"
                    "log10_IC50_uM"
                ): affinity_prediction,
                "IC50_uM": ic50_uM,
                "IC50_M": ic50_M,
                (
                    "binding_affinity_"
                    "likelihood"
                ): affinity_likelihood,
                "affinity_pic50": (
                    affinity_pic50
                ),
            }

            new_row = pd.DataFrame(
                [result_record]
            )

            completed_df = pd.concat(
                [
                    completed_df,
                    new_row,
                ],
                ignore_index=True,
            )

            completed_df.drop_duplicates(
                subset=["dna_sequence"],
                keep="last",
                inplace=True,
            )

            completed_df[
                "_source_order"
            ] = completed_df[
                "dna_sequence"
            ].map(source_order)

            completed_df.sort_values(
                "_source_order",
                inplace=True,
            )

            completed_df.drop(
                columns=["_source_order"],
                inplace=True,
            )

            temporary_output = (
                output_path.parent
                / (
                    output_path.stem
                    + ".temporary.xlsx"
                )
            )

            completed_df.to_excel(
                temporary_output,
                index=False,
            )

            temporary_output.replace(
                output_path
            )

            successful_this_run += 1

            print(
                "Boltz-2 result:"
            )
            print(
                f"  confidence_score = "
                f"{confidence_score}"
            )
            print(
                f"  affinity_likelihood = "
                f"{affinity_likelihood}"
            )
            print(
                f"  affinity_pic50 = "
                f"{affinity_pic50}"
            )
            print(
                f"[SAVED] {len(completed_df)}/60"
            )

        except Exception as error:
            failed_this_run += 1

            print(
                f"[ERROR] {candidate_id}: "
                f"{error}"
            )

            failure_record = pd.DataFrame(
                [
                    {
                        "candidate_id": (
                            candidate_id
                        ),
                        "dna_sequence": (
                            sequence
                        ),
                        "error": str(error)[
                            :1000
                        ],
                    }
                ]
            )

            failure_record.to_csv(
                failure_path,
                mode="a",
                header=(
                    not failure_path.exists()
                ),
                index=False,
            )

        finally:
            if position < len(
                pending_records
            ):
                print(
                    f"Waiting "
                    f"{PER_REQUEST_DELAY_SECONDS} "
                    f"seconds..."
                )

                await asyncio.sleep(
                    PER_REQUEST_DELAY_SECONDS
                )

    print()
    print(
        "=" * 80
    )
    print(
        "Run complete"
    )
    print(
        f"Successful this run: "
        f"{successful_this_run}"
    )
    print(
        f"Failed this run: "
        f"{failed_this_run}"
    )
    print(
        f"Total saved: "
        f"{len(completed_df)}/60"
    )
    print(
        f"Result file: {output_path}"
    )



if __name__ == "__main__":
    asyncio.run(main())
