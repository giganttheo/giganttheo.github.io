---
layout: page
title: Research and Projects
permalink: /research
---

I am currently conducting research on summarization of multimodal presentations, for my PhD at Centrale Supélec / Université Paris-Saclay, under the supervision of Frédéric Dufaux and Camille Guinaudeau.

## Tools

During my PhD, I developed multiple tools to process and study the summarization of multimodal presentations by leveraging their structure. I packaged some of those tools in a [**github repo**](https://github.com/giganttheo/mmmmmm), including methods for slide extraction from presentation records, speech recognition, and the construction of a structured multimodal representation, a cost-effective multimodal input for VLMs.

## Publications

* Théo Gigant, Camille Guinaudeau, Frédéric Dufaux. ***Summarization of Multimodal Presentations with Vision-Language Models: Study of the Effect of Modalities and Structure***. Preprint, 2025, [**[PDF]**](https://arxiv.org/abs/2504.10049)
* Théo Gigant, Camille Guinaudeau, Marc Décombas, Frédéric Dufaux. ***Mitigating the Impact of Reference Quality on Evaluation of Summarization Systems with Reference-Free Metrics***. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, Miami, 2024, [**[PDF]**](https://aclanthology.org/2024.emnlp-main.1078/) [**[blog]**](/evaluating_lm_low_quality_refs) [**[code]**](https://github.com/giganttheo/importance-based-relevance-score)
* Théo Gigant, Frédéric Dufaux, Camille Guinaudeau, Marc Décombas. ***TIB: A Dataset for Abstractive Summarization of Long Multimodal Videoconference Records***. In Proceedings of the 20th International Conference on Content-based Multimedia Indexing, Orléans, 2023, [**[PDF]**](https://dl.acm.org/doi/10.1145/3617233.3617238) [**[blog]**](/tib)

In 2022 I participated in the preprocessing of several datasets as part of the [BigScience Research Workshop](https://bigscience.huggingface.co/), which resulted in two publications:

* Le Scao, Teven, et al. ***Bloom: A 176b-parameter open-access multilingual language model***. 2022, [**[PDF]**](https://inria.hal.science/hal-03850124/)
* Fries, Jason, et al. ***Bigbio: A framework for data-centric biomedical natural language processing.*** In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022, [**[PDF]**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html)

## Other Projects

* [Whisper Medium Romanian](https://huggingface.co/gigant/whisper-medium-romanian) (December 2022): State-of-the-art speech recognition model for Romanian, trained for the [Whisper Fine-Tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event). First in the leaderboard for Romanian ASR.
* [Old Book Illustrations Dataset](https://huggingface.co/datasets/gigant/oldbookillustrations) (July 2022): Collection of 4172 public domain illustrations scanned from old books, collected from the Old Book Illustrations website with their agreement.
* [WikiArt diffusion mini](https://github.com/giganttheo/distill-ccld) (May 2022): As part of HuggingFace's [HugGAN challenge](https://github.com/huggingface/community-events/tree/main/huggan), I worked on a distilled latent diffusion model for text-conditionned image generation trained on the WikiArt dataset of pieces of visual art.
* [Romanian Wav2Vec2](https://huggingface.co/gigant/romanian-wav2vec2) (February 2022): Speech recognition model for Romanian, trained for the [Robust Speech Recognition Challenge](https://discuss.huggingface.co/t/open-to-the-community-robust-speech-recognition-challenge/13614). First in the leaderboard for Romanian ASR.
* [T5-VAE](https://github.com/giganttheo/T5-VAE) (July 2021): Project for the Flax/JAX community week, that combines a T5 transformer model with a variational autoencoder to learn smooth latent spaces for texts.