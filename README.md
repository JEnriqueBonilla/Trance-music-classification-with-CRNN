# Trance music classification with CRNN
This repository contains an exploratory analysis, along with 3 different models, for music label classification. Particularly, 6 important trance music labels were analysed.

## Description

This repository contains an analysis I made for the Data Science conference [The Dataday 2019](https://sg.com.mx/dataday/) and for [the Data Pub](https://www.meetup.com/es/thedatapub/) both in Mexico City, Mexico. The principal model that is used is based on the paper [Musical Artist Classification with Convolutional Recurrent Neural Networks](https://www.researchgate.net/publication/330409573_Music_Artist_Classification_with_Convolutional_Recurrent_Neural_Networks) by Nasrullah, Z. et al. [1] and good part of the functions that are used are based on the code available in the github repository of the paper that can be found [here](https://github.com/ZainNasrullah/music-artist-classification-crnn).

The aim of the analysis is to explore the posibility of training a model that can classify new songs into one out of 6 important trance music labels: Anjunabeats, FSOE, WAO138, Coldharbour recordings, Pure Trance and Armind. Three different models were compare: a multinomial logistic regresion that uses the songs metadata, a naive aproach, i.e. a deep Feedfoward Neural Network and the CRNN used by Nasrullah, Z. et al. [1]. The CRNN showed better results compare to the other two models, given the limitations that the logistic regression has as it uses only the songs metadata and no information of how the actual song sounds like.

## Notebooks

There are two jupyter notebooks in this repository: 

* **CRNN_audio_classificaton_4**: focus on the classification of four labels (Anjunabeats, FSOE, WAO138 and Coldharbour recordings). It contains a brief introduction to the problem, a summary of what is electronic and trance music, an exploratory analysis of the metadata these 4 labels, the results of the models trained, and multiple visualizations of the results.
* **CRNN_audio_classificaton_6** : this notebook has an extension of the model were 6 labels were compared instead of 4.

## Data

The data that was used were all the realized songs from 1st of january 2018 to March 2019 from these 6 labels and that were available on Beatport. All songs that were not label as Trance by Beatport were not used.

As I don't own the rights of these songs the data was not uploaded, just the results.

## Results

One of the main results that were found was that the way CRNN learned how to do the classification shows a possibility in using this information for music recomendation sistems, particularly if you are interested in Djing or you want your customers to explore new labels that are related to the labels they are currently listening.

## Other Links

You can watch the talk I gave in the Data Day 2019 [here](https://www.youtube.com/watch?v=hWZheSer1PM&list=PLnLzwYW6HOC4G5QJ8pWY4WD6dFlGJR8lv&index=13)

## References

[1] Nasrullah, Z. and Zhao, Y., Musical Artist Classification with Convolutional Recurrent Neural Networks. *International Joint Conference on Neural Networks (IJCNN)*, 2019.