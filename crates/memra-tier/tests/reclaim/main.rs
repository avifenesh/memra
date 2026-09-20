//! CPU-only replay of the kv-tier-gate argument and residual-series contracts.
//! The gate's pure modules are included by path: no CUDA, no GPU claim, no G1 verdict.
#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/cli.rs"]
mod cli;
#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/reclaim_contract.rs"]
mod reclaim_contract;

mod day11;
