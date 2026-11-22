#!/bin/bash
# memra model verify checkpoint's native runner for the nvfp4-rig class: the runner contract
# (CKPT_DIR TOKEN..., MEMRA_ORACLE_OUT) with the class named before the checkpoint.
exec "${INK2_GPU:?the ink2-gpu binary}" --class nvfp4-rig "$@"
