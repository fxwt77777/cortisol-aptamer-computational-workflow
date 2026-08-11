FINAL REVISION RESULT PACKAGE
Generated: 2026-08-12

TABLE 1
-------
Table1_candidate_summary.tsv

Contains aptC and final APT1-APT5 sequences and screening metrics.

Important:
- Boltz-2 screening used HCY as a surrogate ligand.
- affinity_pic50 is a Boltz-2-derived screening metric, not an experimentally measured cortisol affinity.
- aptC was scored with the same CNN/Boltz-2 workflow but was not one of the 60 GA-generated candidates; therefore no GA60 rank is assigned.
- APT5 is the final Seq5_mut2_GGGCG sequence, not the original homopolymer-containing Seq5.

TABLE 2
-------
Table2_MD_MMGBSA_summary.tsv
Table2_MD_MMGBSA_replicates.tsv

Contains aptC, APT3 and APT4 independent 3 x 50 ns cortisol MD results.

Important:
- n = 3 independent trajectories.
- Structural summary SD values are calculated across the three trajectory-level means.
- Final MM/GBSA uses the 25-50 ns post-equilibration window.
- MM/GBSA TOTAL is an energy estimate and is not an absolute binding free energy.
- Old 0-50 ns MM/GBSA values are not the final manuscript values.

TABLE 3
-------
Table3_steroid_selectivity.tsv

Contains APT4 cortisol/cortisone/progesterone 10 ns trajectory metrics and 5-10 ns MM/GBSA estimates.

Important:
- Each steroid has one 10 ns trajectory.
- Structural SD values are within-trajectory dispersion, not independent-replicate uncertainty.
- The steroid calculations are exploratory and do not demonstrate clear energetic cortisol selectivity.
- Cortisol, cortisone and progesterone show similar MM/GBSA TOTAL estimates but different interaction modes.

EXCLUDED / HISTORICAL RESULTS
-----------------------------
- Invalid AutoDock Vina results are excluded.
- Historical APT2/APT5 50 ns trajectories were initialized from non-contacting DNA/ligand geometries and are not used for direct bound-start quantitative comparison with aptC/APT3/APT4.
- Historical k1-k6 MM/GBSA sampling sets are not treated as independent replicates.
- Historical 0-50 ns main MM/GBSA summaries are superseded by the final 25-50 ns analysis.
