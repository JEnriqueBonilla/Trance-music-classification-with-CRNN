# Trance music classification with CRNN
This repository contains an exploratory analysis, along with three different models, for music label classification. Mainly, six important trance music labels were analyzed.

## Description

This repository contains an analysis I made for the Data Science conference [The Data day 2019](https://sg.com.mx/dataday/) and [the Data Pub](https://www.meetup.com/es/thedatapub/), both in Mexico City, Mexico. The principal model is based on the paper [Musical Artist Classification with Convolutional Recurrent Neural Networks](https://www.researchgate.net/publication/330409573_Music_Artist_Classification_with_Convolutional_Recurrent_Neural_Networks) by Nasrullah, Z. et al. [1] and a good part of the functions used are based on the code available in the GitHub repository of the paper. That repository can be found [here](https://github.com/ZainNasrullah/music-artist-classification-crnn).

The analysis aims to explore the possibility of training a model that can classify Trance music songs into one out of six essential Trance labels: Anjunabeats, FSOE, WAO138, Coldharbour recordings, Pure Trance and Armind. Three different models were compared: a multinomial logistic regression that uses audio metadata such as song key, BPM, and length; a naive approach, i.e., a deep Feedforward Neural Network, and the CRNN used by Nasrullah, Z. et al. [1]. The CRNN showed, as expected, better results considering the limitations that logistic regression has in the context of audio classification.

## Notebooks

There are two Jupyter notebooks in this repository: 

* **CRNN_audio_classificaton_4** focuses on the classification of four labels: Anjunabeats, FSOE, WAO138 and Coldharbour recordings. It contains an introduction to the problem, a summary of what electronic and trance music are, an exploratory analysis, the results of the trained models, and multiple visualizations of the results.
* **CRNN_audio_classificaton_6**  has an extension of the previous analysis, but now six labels were compared instead of four. 

## Data

All songs from January 2018 to March 2019 on Beatport are considered for each of the six labels. However, all the songs that were not labelled as Trance by Beatport were discarded. As I don't own the rights of these songs, the data was not uploaded, just the results.

## Results

The CRNN showed a good classification score given the high variability of sounds that a label can have and that several songs could belong to more than one label. Furthermore, the t-SNE visualizations showed a possibility of using the network learnings for music recommendation systems. Notably, it can be used if you are interested in Djing (which song could I play after this one?), or you want your customers to explore new music labels that are related to the ones they are currently listening to or the ones they like the most.

## Other Links

You can watch the talk I gave in the Data Day 2019 [here](https://www.youtube.com/watch?v=hWZheSer1PM&list=PLnLzwYW6HOC4G5QJ8pWY4WD6dFlGJR8lv&index=13).

## References

[1] Nasrullah, Z. and Zhao, Y., Musical Artist Classification with Convolutional Recurrent Neural Networks. *International Joint Conference on Neural Networks (IJCNN)*, 2019.