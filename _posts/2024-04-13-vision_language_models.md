---
layout: post
title: "The state of Vision Language Models"
author: Théo Gigant
use_math: true
category: multimodal nlp
image: holzer.jpg
---

*Illustration: Truisms (Jenny Holzer, 1977–79)*

Vision and language models are the new shiny thing in the *AI* space. [Some](https://palm-e.github.io/) are big, [some](https://huggingface.co/openbmb/MiniCPM-V-2) are small, [some](https://arxiv.org/abs/2404.04346) are very complex machinery, [some](https://www.adept.ai/blog/fuyu-8b) are as simple as it gets, [some](https://llava-vl.github.io/) can only understand images, [some](https://arxiv.org/abs/2402.08268) whole hour-long videos, [others](https://arxiv.org/abs/2401.10208) can also generate images.

One thing we can interpret from all these different models is the choices that were made and the results they yield. Especially, in this blog post we will focus on the automatic understanding of vision and language by describing some of the popular designs that were studied in the recent developments of Vision-Language Models.

## Vision and language in a shared latent space

[CLIP](https://arxiv.org/abs/2103.00020) is a simple but effective framework that jointly learns a vision and a text encoder, trained to project images and captions in a shared latent space in which an image is close to its caption.

![](https://i.ibb.co/TPZ69BY/clip.png)

*Illustration: CLIP contrastive pre-training ([OpenAI Blog](https://openai.com/research/clip))*

It is a building block of most multimodal vision-language models: *eg* the text encoder in text-conditionned image generation such as all the [Stable Diffusion models](https://stability.ai/stable-image), or the image encoder in language and vision chatbots such as [LLaVA](https://llava-vl.github.io/).

In the frameworks that aim at understanding language and vision, the ViT image encoder from CLIP (or newer CLIP-inspired techniques such as [SigLIP](https://arxiv.org/abs/2303.15343)) is often used as the vision backbone.

A key advantage is that the latent tokens representations in CLIP's ViT might have some sort of a cross-modal / *[synaesthetic](https://arxiv.org/abs/2306.03678)* ability, by already being *mostly* aligned with their captions.

*"Mostly"*, because the latent representation of the image is aligned to the latent representation of the text, that went through the tokenizer and the transformer-encoder, while in most scenarios the encoded image is fed to a language model along freshly-embedded text tokens.

In order to re-align visual tokens with the text tokens, and, optionnaly, compress, focus or select the visual information that will be forwarded to the language model, the encoded image tokens are processed by a "Visual Abstractor" model.

## Leveraging and aligning pretrained models with a "Visual Abstractor"

When using the image encoder from CLIP, the images are mostly pre-aligned with text and we could just map the CLIP latents to the text token embeddings, with a minimalistic projection layer that will be trained on image/caption pairs. This is the idea behing the [LLaVA](https://llava-vl.github.io/) framework.

![](https://llava-vl.github.io/images/llava_arch.png)
*Illustration: LLaVA architecture ([LLaVA Blog](https://llava-vl.github.io/))*

They call this mapping the "projection", and it is trained on image/caption pairs while keeping the vision and language models frozen. This projection and the language model are tuned during "visual instruction tuning", a second, more expensive, training stage aimed at teaching the model to follow instructions on visual tasks.

In the first LLaVA, this abstractor was as simple linear projection. In consequent versions (LLaVA 1.5 and 1.6/NeXT), it was swapped for a more expressive Multi-Layer Perceptron (MLP).

While minimalistic and effective, this "projection" strategy has the default of keeping the number of tokens from the encoded image, *ie* $16*16=256$ tokens with ViT. For some applications --say video understanding-- the total number of tokens might blow up, and be very redundant too. In such situations, a "Visual Abstractor" can select the information from a varying number of images with a fixed tokens budget, with popular choices being the Q-Former ([BLIP-2](https://arxiv.org/abs/2301.12597)) or the Perceiver Resampler ([Flamingo](https://arxiv.org/abs/2204.14198)) abstractors. Both are using learnt queries and attention to select the salient visual information for a given token budget, but Q-Former is also conditionned on input text.

[*Cha et al*](https://arxiv.org/abs/2312.06742) studied other visual abstractor strategies more in-depth, based on convolution neural networks (C-Abstractor), or deformable attentions (D-Abstractor), along adaptive average pooling which allows to select the number of output tokens.

[*Li et al*](https://arxiv.org/abs/2311.17043) proposed to only keep two tokens for each frame for video understanding: one that only encode the frame information (dubbed "content" token), and another one, conditionned on input text, aiming to encode the contextualized information (dubbed "context" token).

All these ideas rely on aligning and filtering multiple pretrained models to leverage their multimodal capabilities.

## Multimodal fusion: are images a foreign language?

As shown empirically by the [ViT](https://arxiv.org/abs/2010.11929) model, images can be processed with the same architecture as text, with state-of-the-art performance. The image is split into patches, that are embedded and processed by a language model as if they were text tokens. Effectively, an image becomes a foreign language, and *Wang et al* tested it quite litteraly. Their [BeiT 3](https://arxiv.org/abs/2208.10442) model follows the ViT architecture with a multimodal twist, as the model is trained from scratch with image and text tokens processed in the same model but with different experts.

Halfway between aligning pretrained models and training a model with all modalities, falls Adept's [Fuyu](https://www.adept.ai/blog/fuyu-8b) framework. They simplified both the architecture and training procedure by feeding the image patch embeddings as is to a language model. With that framework, there is no need to think about how to scale the vision encoder vs the language model, or what training stages to do and in what order, and the model is able to work with images of varying resolutions. This last particularity was then improved upon by [*Li et al*](https://arxiv.org/abs/2311.04219) in their OtterHD model.

![](https://www.adept.ai/images/blog/fuyu-8b/architecture.png)
*Illustration: Fuyu architecture ([Adept Blog](https://www.adept.ai/blog/fuyu-8b))*

The authors claim that the Fuyu framework is "*easier to understand, scale, and deploy*", but give no information about the amount of data used and the cost for training such model. It would be no surprise if it is orders of magnitude more than with the LLaVA framework, for comparable results.

## *ARE* images a foreign language?

An aspect we might reflect on is the granularity of modalities.

An earlier work on multimodal models by [*Alayrac et al*](https://arxiv.org/pdf/2006.16228.pdf) proposed to merge the modalities at different points depending on their granularity. Audio and vision are treated as fine-grained, while text is more coarse-grained.

The paper reads:

> This strategy is based on the observation that the visual and audio spaces are fine-grained (there are many visual or sounds of guitars that might be really different to each other) while the textual domain is more coarse as its goal is to abstract away details (e.g. a single “guitar” word).

This idea weighs in favor of pre-processing the images first, *eg* by using an image encoder before feeding the resulting embeddings to the Language Model.

However, are we sure to know how finer-grained vision is, compared to text? And do all text tokens have the same granularity?

One might argue that some words with lots of different meanings depending on the context have a different granularity compared to *stop-words* for instance. From my understanding, an example that might be interpreted in this direction, is the recent discovery by [*Raposo et al*](https://arxiv.org/abs/2404.02258) that all tokens don't need the same model depth.

All visual tokens are not as fine-grained as well, with the example of documents vs real-world pictures.

Maybe in this situation, the better solution will be to just throw everything at once in the model, and let it figure out how much processing each token need.

## Where are Vision-Language Models headed?

Obviously, we cannot know for sure. But there is something that we can take from [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html): all these tricks to cleverly leverage and align pretrained models of different modalities, and to filter and focus the visual content for a given token budget, are just temporary solutions. At some point, we might have models trained end-to-end to figure everything by themselves based on statistics of more-and-more massive datasets. *eg* [Fuyu](https://www.adept.ai/blog/fuyu-8b)-style with infinite-context.

For the time being, we better be thoughtful and deliberate in our choices, in order to create useful vision-language models for different tasks with limited training budgets and computations.