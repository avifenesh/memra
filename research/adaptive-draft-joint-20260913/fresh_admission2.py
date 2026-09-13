"""Second fresh test with the dense checkpoint and normal probe cadence16."""
import fresh_admission as experiment
experiment.OUT=experiment.ROOT/'raw/fresh-admission2'
experiment.CHECKPOINT=experiment.ROOT/'checkpoints/admission-dense-model.json'
experiment.PROMPTS=experiment.ROOT/'checkpoints/fresh-admission2-prompts'
experiment.DONE=experiment.ROOT/'FRESH_ADMISSION2_DONE'
experiment.main()
