# ASR_NeMO

Model currently selected is RIVA Quartznet ASR English. More information can be found at: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_en_us_quartznet 

'nsc corpus training.ipynb' is a Juptyer notebook to test out fine-tuning of the Quartznet model. The dataset used is from the NSC Singapore English Corpus Part 1. The model was trained with 23871 lines, validated with 2373 lines and tested with 2373 lines. The out-of-the-box model had a word error rate of 0.0900. Training was done by freezing the encoder and unfreezing the batch normalisation, running for a total of 2 epochs in 636.485 seconds. After training, the word error rate increased to 0.0944 instead.
