//! CPU-only replay of the kv-tier-gate argument, residual-series and fault-arm contracts.
//! The gate's pure modules are included by path: no CUDA, no GPU claim; the series verdict
//! and the fault verdicts are replayed from committed receipt bytes, never produced here.
#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/cli.rs"]
mod cli;
#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/fault_contract.rs"]
mod fault_contract;
#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/reclaim_contract.rs"]
mod reclaim_contract;

mod day11;
mod day12;
mod fault;
