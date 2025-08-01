#!/usr/bin/env python3
"""
Simple msprime + stdpopsim simulation for validation
Generates 2 populations with known pi, dxy, fst values + VCF output
"""

import stdpopsim
import msprime
import argparse
import pandas as pd
import os
import numpy as np
import pyfastx
import time
from loguru import logger
from snakemake_argparse_bridge import snakemake_compatible


def read_fasta(fasta: str) -> str:
    """
    reads fasta and returns seq of first entry.
    """
    for _, seq in pyfastx.Fasta(fasta, build_index=False):
        if isinstance(seq, str):
            return seq
        else:
            raise TypeError(f"Expected string sequence, got {type(seq)}")
    
    raise ValueError(f"No sequences found in {fasta}")

@snakemake_compatible({"outdir":"params.outdir", "samples_per_pop":"params.num_samples", "reference_fasta":"input.ref", "seed":"params.seed"})
def main():
    parser = argparse.ArgumentParser(description="Simple 2-population simulation")
    
    parser.add_argument(
        "--outdir", default="simulation_output", help="Output directory"
    )
    parser.add_argument("--samples", default=5, dest="samples_per_pop", type=int)
    parser.add_argument(
        "--ref", dest="reference_fasta", help="path to ref fasta"
    )

    parser.add_argument("--seed", dest="seed", help="random seed")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger.add(os.path.join(args.outdir, "log.txt"))
    species = stdpopsim.get_species("DroMel")
    contig = species.get_contig("2L", left=1e6, right=2e6)

    reference_seq = read_fasta(args.reference_fasta)
    logger.info(f"Read fasta: {args.reference_fasta}, found {len(reference_seq)}nt")

    Ne1 = 1720600

    total_samples = args.samples_per_pop * 2

    logger.info(f"Simulating ancestry {total_samples=}")

    start_time = time.time()
    ts = msprime.sim_ancestry(
        samples=total_samples,  # 20 haploid samples is s=5
        population_size=Ne1,
        sequence_length=len(reference_seq),
        recombination_rate=0,
        random_seed=args.seed,
    )
    ancestry_time = time.time() - start_time
    logger.info(f"Ancestry simulation completed in {ancestry_time:.2f} seconds")

    logger.info("Adding mutations...")
    start_time = time.time()
    mutation_rate = species.genome.mean_mutation_rate
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=args.seed)
    mutation_time = time.time() - start_time
    logger.info(f"Mutation simulation completed in {mutation_time:.2f} seconds")

    logger.info(f"Generated {ts.num_sites} variant sites")
    logger.info(f"Number of samples: {ts.num_samples}")
    logger.info(f"Number of populations: {ts.num_populations}")

    logger.info("Calculating true statistics...")

    # Pop1: first args.samples_per_pop * 2 haplotypes (0 to 9 if samples_per_pop=5)
    # Pop2: next args.samples_per_pop * 2 haplotypes (10 to 19 if samples_per_pop=5)

    pop1_samples = list(range(args.samples_per_pop * 2))
    pop2_samples = list(range(args.samples_per_pop * 2, args.samples_per_pop * 4))

    pi_pop1 = ts.diversity(sample_sets=[pop1_samples])
    pi_pop2 = ts.diversity(sample_sets=[pop2_samples])
    pi_total = ts.diversity()

    dxy = ts.divergence(sample_sets=[pop1_samples, pop2_samples])
    fst = ts.Fst(sample_sets=[pop1_samples, pop2_samples])

    pi_pop1 = float(pi_pop1)
    pi_pop2 = float(pi_pop2)
    pi_total = float(pi_total)
    dxy = float(dxy)
    fst = float(fst)

    logger.info("\n=== TRUE VALUES ===")
    logger.info(f"π Pop2:     {pi_pop2:.6f}")
    logger.info(f"π Total:    {pi_total:.6f}")
    logger.info(f"π Pop1:     {pi_pop1:.6f}")
    logger.info(f"dxy:        {dxy:.6f}")
    logger.info(f"Fst:        {fst:.6f}")

    expected_pi = 4 * Ne1 * mutation_rate
    expected_dxy = 4 * Ne1 * mutation_rate
    expected_fst = (expected_dxy - expected_pi) / expected_dxy

    logger.info("\n=== EXPECTED VALUES ===")
    logger.info(f"π Expected:   {expected_pi:.6f}")
    logger.info(f"dXY Expected: {expected_dxy:.6f}")
    logger.info(f"FST Expected: {expected_fst:.6f}")

    truth_df = pd.DataFrame(
        [
            {
                "pi_pop1": pi_pop1,
                "pi_pop2": pi_pop2,
                "pi_total": pi_total,
                "dxy": dxy,
                "fst": fst,
                "mutation_rate": mutation_rate,
                "Ne1": Ne1,
                "expected_pi": expected_pi,
                "expected_dxy": expected_dxy,
                "expected_fst": expected_fst,
            }
        ]
    )

    truth_df.to_csv(os.path.join(args.outdir, "truth.csv"), index=False)
    logger.info(f"\nSaved truth values: {os.path.join(args.outdir, 'truth.csv')}")

    sample_names = [f"Pop1_sample_{i}" for i in range(args.samples_per_pop)] + [
        f"Pop2_sample_{i}" for i in range(args.samples_per_pop)
    ]

    ts.dump(os.path.join(args.outdir, "sim.trees"))
    logger.info(f"Saved tree sequence: {os.path.join(args.outdir, 'sim.trees')}")

    pop_map = pd.DataFrame(
        {
            "sample": sample_names,
            "population": ["Pop1"] * args.samples_per_pop
            + ["Pop2"] * args.samples_per_pop,
        }
    )
    pop_map.to_csv(
        os.path.join(args.outdir, "popmap.txt"), sep="\t", index=False, header=False
    )
    logger.info(f"Saved population map: {os.path.join(args.outdir, 'popmap.txt')}")

    logger.info("Writing sample alignments...")
    start_time = time.time()
    os.makedirs(os.path.join(args.outdir, "alignments"), exist_ok=True)
    alignments = list(ts.alignments(reference_sequence=reference_seq))
    logger.info(f"{len(alignments)=}")

    # alignments 0-9: Pop1 (individuals 0-4, haplotypes 0-9)
    # alignments 10-19: Pop2 (individuals 0-4, haplotypes 10-19)

    for i in range(0, len(alignments), 2):
        haplotype_pair_id = i // 2  # 0,1,2,3,4,5,6,7,8,9
        hap1, hap2 = alignments[i], alignments[i + 1]

        if haplotype_pair_id < args.samples_per_pop:
            # First args.samples_per_pop individuals belong to Pop1
            sample_name = f"Pop1_sample_{haplotype_pair_id}"
        else:
            # Next args.samples_per_pop individuals belong to Pop2
            sample_name = f"Pop2_sample_{haplotype_pair_id - args.samples_per_pop}"

        # Write haplotype 0
        with open(
            os.path.join(args.outdir, "alignments", f"{sample_name}.fasta"), "w"
        ) as f:
            f.write(f">{sample_name}_hap0\n")
            f.write(hap1 + "\n")

            # Write haplotype 1
            f.write(f">{sample_name}_hap1\n")
            f.write(hap2 + "\n")

    haplotype_time = time.time() - start_time
    logger.info(f"Haplotype writing completed in {haplotype_time:.2f} seconds")

    tables = ts.dump_tables()

    # Track existing variant positions
    existing_pos = np.round(ts.tables.sites.position).astype(int)
    existing_pos_set = set(existing_pos)

    # Add sites for invariant positions
    for pos in range(int(ts.sequence_length)):
        if pos not in existing_pos_set:
            # Optional: derive ancestral state from reference
            tables.sites.add_row(
                position=pos, ancestral_state="."
            )  # or use "." if you don't care

    # Re-sort and convert back to tree sequence
    tables.sort()
    ts_all = tables.tree_sequence()
    with open(os.path.join(args.outdir, "perfect.vcf"), "w") as f:
        ts_all.write_vcf(
            f,
            position_transform=lambda x: 1 + np.round(x),
            individual_names=sample_names,
        )

if __name__ == "__main__":
    main()
