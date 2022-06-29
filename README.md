# Testing of GPU usage with Singapore Corpus 

'nsc_corpus_training.ipynb' is a Jupyter notebook to test out fine-tuning of the Quartznet model. The dataset used is from the NSC Singapore English Corpus Part 1. The model was trained with 23871 lines, validated with 2373 lines and tested with 2373 lines. The out-of-the-box model had a word error rate of 0.089951. Training was done by freezing the encoder and unfreezing the batch normalisation, running for a total of 10 epochs in 4237.57 seconds with 1 GPU utilised. After training, the word error rate increased to 0.148990 instead.

'nsc_corpus_training.py' is a Python script to test out the utilisation of 4 GPUs in training. Training and validation batch size was changed to 16 to prevent 'CUDA error: out of memory'. For the same number of epochs and the same training and validation dataset, time taken for training was 1760.28414 seconds. 

# Fine-tuning Pre-trained Model with Location Data

Model currently selected is RIVA Quartznet ASR English. More information can be found at: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_en_us_quartznet 

'location_training_test1_encoder.ipynb' involves fine-tuning the model with 9 audio files, with a total duration of 0.01 hours, ran for a total of 50 epochs. The new dataset consists of locations, namely bishan, boon lay, bukit batok, bukit gombak, jurong, sembawang, tiong bahru and tuas. The audio files are resampled at 16kHz before training. Training the model without freezing the encoder results in more accurate location predictions. However, when tested with the original NSC Singapore Corpus, the WER increased to 0.729195. Training the model with a frozen encoder results in less accurate location predictions but the WER of the original NSC Singapore Corpus also increased significantly to 0.519756.  
**Conclusion: Size of dataset needs to increase.**

'location_training_test2_augment_optimise.ipynb' involves fine-tuning the model with 177 audio files for training and 10 audio files for validation, with a total duration of 0.16 hours, ran for a total of 50 epochs. The audio files are resampled at 16kHz and cleaned with the noisereduce module before training. Parameters in the optimiser and spectrogram augmentation was adjusted according to NVIDIA tutorial for language finetuning, found here: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb The encoder layer was also frozen before training. However, the WER of the original NSC Singapore Corpus continued to increase significantly after training with the new dataset, from 0.04537 to 0.9998491. 
**Conclusion: NVIDIA's tutorial cannot be applied, possibly because the new dataset is too small. More parameters and layers have to be frozen.**

'location_training_test3_batchnorm.ipynb' involves fine-tuning the model with 177 audio files for training and 10 audio files for validation, with a total duration of 0.16 hours, ran for a total of 2 epochs. The audio files are resampled at 16kHz and cleaned with the noisereduce module before training. To prevent overfitting of the model, both the encoder and decoder were frozen, with the batch normalisation layers unfrozen. The WER of the original NSC Singapore Corpus increased, but less significantly in comparison, from 0.04537 to 0.134325 after training. Testing on the location training dataset, the WER also decreased from 1.26931 to 0.861386. Not captured in the WER, for the same audio file for bukit gombak was predicted as 'but get come back' using the untrained model and predicted as 'buki komba' using the trained model. Such a prediction will still contribute to the WER, but the locations predicted are now closer to the transcripts and easier for the user to decode. **Conclusion: continue to freeze encoder and decoder during training. Adjust optimisation and spectrogram augmentation parameters. Possibly introduce language models for 'buki komba' to be outputted as 'bukit gombak'**

'location_training_test4_traintestval.ipynb' has the same functionality as 'location_training_test3_batchnorm.ipynb', but with a proper separate unseen location dataset for testing instead of reusing the training dataset. It has 177 audio files for training, 16 files for validation and 16 files for testing. The WER of the original NSC Singapore Corpus increased from 0.04537 to 12.376 after training. Testing on the location training dataset, the WER also decreased from 1.15217 to 0.913304.

# Implementing ASR on Botpress 

'speechToTextv1 is a chrome extension built that uses the Web Speech API to transcribe speech. The transcript will appear in a small textbox on the bottom right of the screen and in any textarea element. Since the 

# Using TTS FastPitch Model for Synthetic Voice Generation
'fastPitch_finetuning.ipynb' uses the NSC Corpus to generate a synthetic voice model for each speaker. 

