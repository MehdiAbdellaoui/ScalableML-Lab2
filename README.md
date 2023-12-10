# ScalableML-Lab2

## Model improvements

### Model centric improvements
With significantly more time and less limited computing resources a hyperparameter search can be done in order to find the set which gives the best performance. Since we were limited in resources we instead opted to apply dropout in order to improve the model performance. Specifically we chose to use a dropout rate of 15% when training the model.

### Data centric improvements
We only used the data from the mozilla-foundation/common_voice_11_0 dataset. The realiability of the model could be improved in a few ways. We could make sure to include many different dialects. We could also include noisy data by either using a data set with noisy data or adding noise to our own dataset. It could also be a good idea to simply exend the dataset with more samples.

### Results

We did three training runs on local hardware. The first time with batch size 4. The second time with batch size 4 and dropout. The third time with batch size 6 and dropout. Due to hardware limitation we could not use batch size 8 or above.

| Parameters | Eval loss| Eval wer |
|----------|:-------------:|------:|
| batch size 4, no dropout | 0.290 | 63.8 |
| batch size 4, dropout | 0.289 | 60.8 |
| batch size 6, dropout | 0.280 | 46.6 |

## Feature pipeline

The feature pipeline dowloads the dataset from huggingface mozilla-foundation/common_voice_11_0. It then removes all columns except for audio and transcript. Once that is done it uses the whisper feture extractor and tokenizer on these fields to transform them into data which whisper can use. Finally we upload our processed data onto huggingface which we use as the feature store.

## Training pipeline

The training pipeline downloads our processed data set and a processor from huggingface and then creates a data collator. It then defines the wer metric to use for testing and downloads a pretrained whisper model. Finally we define training parameters and start the training. The training is done in 4000 steps and use checkpoints uploaded to huggingface.

## Inference pipeline

Huggingface space: link.com
