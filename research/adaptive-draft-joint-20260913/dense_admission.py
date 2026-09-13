"""Densify old training/calibration only; never reuse the failed fresh test."""
import shutil
import fresh_admission as experiment

experiment.OUT = experiment.ROOT/'raw/dense-admission'
experiment.PROMPTS = experiment.ROOT/'checkpoints/dense-admission-prompts'
experiment.DONE = experiment.ROOT/'DENSE_ADMISSION_DONE'
experiment.EXPECTED_PROMPTS = 32
experiment.EXTRA_CONFIG = {'MEMRA_GEMMA_PROBE_EVERY':'1'}
experiment.PROMPTS.mkdir(exist_ok=False)
for path in (experiment.OLD/'raw/row-oracle/prompts').glob('*.txt'):
    if path.name.startswith(('train-','calibration-')):
        shutil.copyfile(path, experiment.PROMPTS/path.name)
experiment.main()
