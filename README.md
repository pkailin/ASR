# ASR_NeMO

Model currently selected is RIVA Quartznet ASR English. More information can be found at: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_en_us_quartznet 

'nsc_corpus_training.ipynb' is a Jupyter notebook to test out fine-tuning of the Quartznet model. The dataset used is from the NSC Singapore English Corpus Part 1. The model was trained with 23871 lines, validated with 2373 lines and tested with 2373 lines. The out-of-the-box model had a word error rate of 0.089951. Training was done by freezing the encoder and unfreezing the batch normalisation, running for a total of 10 epochs in 4237.57 seconds with 1 GPU utilised. After training, the word error rate increased to 0.148990 instead.

'nsc_corpus_training.py' is a Python script to test out the utilisation of 4 GPUs in training. Training and validation batch size was changed to 16 to prevent 'CUDA error: out of memory'. For the same number of epochs and the same training and validation dataset, time taken for training was 1760.28414 seconds. 
