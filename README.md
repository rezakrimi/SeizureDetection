# Description
This is the source code for my research on detecting epileptic seizures in real-time. Our results has been published in our paper which has been accepted at [ISCAS 2020](https://www.iscas2020.org/) and will be presented on October.
In this work, we used the very well-known [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) database which includes signal recordings from 23 patients and the start and the end of each seizure is labeld by an epileptic seizure expert.
This research was first presented at [York University's Undergraduate Research Conference 2019](http://www.lassondeundergraduateresearch.com/past-researchers) where it was chosen as one of the top 5 research projects to have an oral presentation among ~70 other projects. It also won the first plac for the best oral presentation of the conference. The slides from the presentation can be found at http://rezakrimi.com/projects.

# Files
__feature_extraction.py:__ Takes the raw signal recordings from each patients and extracts the signal energy and PLV value of for each `t` second window of signal.
__solo_training.py & pool_training.py:__ These are the main training scripts which will take the feature vectors for each patient and trains the seizure detection model for them. The only difference between the two is that `pool_training.py` can take advantage of multi-thread model training.
__PLV_visualization & energy_visualization:__ These two scripts visualize the data in different ways. Some of these visualizations can be seen below or on the slides mentioned in the description.
![FFT Spectrogram](https://raw.githubusercontent.com/rezakrimi/SeizureDetection/master/FFT_Spectrogram.png?raw=true)
![PLV & Energy](https://github.com/rezakrimi/SeizureDetection/blob/master/PLV%26Energy.png?raw=true)
