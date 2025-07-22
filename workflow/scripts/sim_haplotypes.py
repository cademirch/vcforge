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
import pyfastx
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

@snakemake_compatible({"outdir":"params.outdir", "samples_per_pop":"params.num_samples", "reference_fasta":"input.ref"})
def main():
    parser = argparse.ArgumentParser(description="Simple 2-population simulation")
    
    parser.add_argument(
        "--outdir", default="simulation_output", help="Output directory"
    )
    parser.add_argument(
        "--samples", default=5, dest="samples_per_pop"
    )
    parser.add_argument(
        "--ref", dest="reference_fasta", help="path to ref fasta"
    )
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logger.add(os.path.join(args.outdir, "log.txt"))
    species = stdpopsim.get_species("DroMel")
    contig = species.get_contig("2L", left=1e6, right=2e6)

    reference_seq = read_fasta(args.reference_fasta)
    logger.info(f"Read fasta: {args.reference_fasta}, found {len(reference_seq)}nt")

    demography = msprime.Demography()

    Ne1 = 10000  
    Ne2 = 10000  
    T_split = 2000  
    migration_rate = 1e-5 

    demography.add_population(name="Pop1", initial_size=Ne1)
    demography.add_population(name="Pop2", initial_size=Ne2)
    demography.add_population(name="Ancestral", initial_size=15000)
    demography.add_population_split(
        time=T_split, derived=["Pop1", "Pop2"], ancestral="Ancestral"
    )

    demography.add_migration_rate_change(
        time=0, rate=migration_rate, source="Pop1", dest="Pop2"
    )
    demography.add_migration_rate_change(
        time=0, rate=migration_rate, source="Pop2", dest="Pop1"
    )
    demography.sort_events()
    logger.info(
        f"Demographic model: Ne1={Ne1}, Ne2={Ne2}, T_split={T_split}, migration={migration_rate}"
    )

    logger.info("Simulating ancestry...")
    ts = msprime.sim_ancestry(
        samples={"Pop1": args.samples_per_pop, "Pop2": args.samples_per_pop},
        demography=demography,
        sequence_length=len(reference_seq),
        recombination_rate=contig.recombination_map.mean_rate,
    )

    logger.info("Adding mutations...")
    mutation_rate = species.genome.mean_mutation_rate
    ts = msprime.sim_mutations(ts, rate=mutation_rate)

    logger.info(f"Generated {ts.num_sites} variant sites")

    logger.info("Calculating true statistics...")

    pop1_samples = list(range(args.samples_per_pop))
    pop2_samples = list(range(args.samples_per_pop, 2 * args.samples_per_pop))

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
                "Ne2": Ne2,
                "T_split": T_split,
                "migration_rate": migration_rate,
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
    pop_map.to_csv(os.path.join(args.outdir, "popmap.txt"), sep="\t", index=False)
    logger.info(f"Saved population map: {os.path.join(args.outdir, 'popmap.txt')}")

    
    
    logger.info("Writing sample alignments...")
    os.makedirs(os.path.join(args.outdir, "alignments"), exist_ok=True)
    alignments = list(ts.alignments(reference_sequence=reference_seq))
    logger.info(f"{len(alignments)=}")
    for i in range(0, len(alignments), 2):
        individual_id = i // 2
        hap1, hap2 = alignments[i], alignments[i + 1]
        
        if individual_id < args.samples_per_pop:
            sample_name = f"Pop1_sample_{individual_id}"
        else:
            sample_name = f"Pop2_sample_{individual_id - args.samples_per_pop}"
        
        with open(os.path.join(args.outdir, "alignments", f"{sample_name}.fasta"), "w") as f:
            f.write(f">{sample_name}_hap0\n")
            f.write(hap1 + "\n")
            f.write(f">{sample_name}_hap1\n")
            f.write(hap2 + "\n")
    
    logger.info(f"Saved {len(sample_names)} alignment files to {os.path.join(args.outdir, 'alignments')}")


if __name__ == "__main__":
    main()
