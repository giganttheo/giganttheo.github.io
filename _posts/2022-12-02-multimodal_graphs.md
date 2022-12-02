---
layout: post
title: "Multi-Modal Graphs: modeling complex structures"
author: Théo Gigant
use_math: true
category: multimodal graph
image: catalan-landscape.jpg
---

*Illustration: The Hunter (Catalan Landscape) (Joan Miró, 1924)*

[*A Gentle Introduction to Graph Neural Networks*](https://distill.pub/2021/gnn-intro/) reads:
>Graphs are all around us; real world objects are often defined in terms of their connections to other things. A set of objects, and the connections between them, are naturally expressed as a *graph*.

### Graphs describe and model complex systems

Graphs are mathematical structures, used to represent entities and their relations.

A graph $G = (V,E)$ is defined by a set of nodes $V$ and a set of edges $E \subseteq \\{ (x,y) \vert (x,y) \in V^2, x \neq y \\}$ that connect them.

Nodes represent entities, and the edges are connecting pairs of nodes to indicate the relation between them. Nodes and edges can have features to encode information about them. Edges can also be directed to capture asymmetric relations.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVVYJclDBmKnQWPKYsw4IU2aJ4AIySSusuCw&usqp=CAU)

*An undirected graph with 6 nodes (labeled 0 to 5) and 8 edges*

A graph can be represented with an adjacency matrix $(n_{nodes}, n_{nodes})$ with $n_{nodes} = \vert V \vert$. The elements of the matrix indicate whether pairs of vertices are connected with an edge or not.

Usually $\vert V \vert ^2 \gg \vert E \vert$ so this matrix is very sparse.

In Machine Learning applications it is efficient to represent a graph with an adjacency list, *ie* a list of pairs of vertices that are connected with an edge.

For instance in the [`jraph`](https://github.com/deepmind/jraph) library, a graph is defined in the class `GraphsTuple` with:
* `nodes`: the node features
* `senders` and `receivers`: lists of ids of nodes that are connected with a directed edge from sender to receiver
* `edges`: the edge features
* `n_node`: number of nodes
* `n_edge`: number of edges
* `globals`: global feature (*eg* graph label)

Graphs are a structure that conveniently model complex systems, such as multi-modal documents.

As an example, in *Extractive Text-Image Summarization with Relation-Enhanced Graph Attention Network*, Xie *et al* construct this graph to model an article that contains texts and images:

![](https://i.ibb.co/Tb8dXdG/regatsum.png)

### Structured representation of documents

There are multiple ways a document can be represented. A textual document can non-exhaustively be represented as:
* A bag (bag of words: BOW): only contains the words with no structural information
* A sequence: contains the words and the order of the words
* A graph: contains the words and relationships between them

The graph representation is the most general, and the most powerful, as sequence and bag are specific graphs.

* The **bag** representation is simple to create and store and it allows to attend to all the words at the same time. This representation is really convenient for training because it is parallelizable and order structure can be recovered by using positional encoding. It is the representation used in the Transformer model for instance. However this representation is computationally expensive for large documents as it forces to attend to the whole document at the same time. Attention layers for instance use this representation and have a quadratic complexity over the size of the input, which is not ideal for scaling to large documents.
* The **sequence** representation is also simple to create and store and it contains minimal structural information as it only keeps the order / position information. Models that use this representation, such as recurrent neural networks, usually have a linear complexity over the size of the input. However this representation suggests to attend words in order and is inconvenient to model long range relationship.
* The **graph** representation is trickier to create and store, but it contains as much structural information as one wants through all the options offered while constructing the graph. It is convenient to model long range relationship, while being sparser than the bag representation. Graph Neural Networks are a class of models designed to deal with documents using graph representations.

Even for textual documents, graph representation come with a lot of advantages. Even though, as of 2022, Bag-Of-Words representation (through the Transformer architecture) is the *de facto* standard in Natural Language Processing tasks, graph representation is receiving a lot of interest from the NLP community: as shown in Wu *et al*'s [survey](https://arxiv.org/abs/2106.06090) on Graph Neural Networks for NLP.

For multi-modal documents, *ie* documents that contain more than one modality (such as text / image / audio), the structural information is richer than in textual documents and cannot always be represented as sequential.

**Graphs make a lot of sense for representing such data.**

To illustrate this, we will construct a simple graph representation of a videoconference record, by taking advantage of the structure of the data.

We will assume that the records consist of a shared-screen video stream and an audio stream, that are being preprocessed so that we have:
* the slides as images and the time frames for which they appear on screen
* the textual transcript of the audio, with the time frames for which each sentence is being said

With only this information, we can construct this graph:

![](https://i.ibb.co/1JY0Q3X/graph-mm.png)

Nodes are of 3 types:
* `slide nodes` that are the still images from the screen-share stream
* `word nodes` that are the words from the transcript
* `sentence nodes` that are the sentences from the transcript

Edges are of 3 types:
* `contiguous` that represent the sequential relationships: a word/sentence/slide and the following/previous one
* `simultaneous` that represent relationships between audio and visual happening at the same time
* `in` that represent relationships between words and the sentence there are in


Node (respectively edge) types are encoded in the node (resp. edge) features

This representation contains more structural information than a multimodal bag of tokens for instance while being sparser.

Typically, a videoconference record is ~1 hour long, and, when tokenized, vastly exceeds the size limit of Transformer models. There are Transformer variants that are designed specifically for longer inputs: the input size limit is `4096` tokens for the [BigBird](https://arxiv.org/abs/2007.14062) model, and `2048` tokens for the [Perceiver](https://arxiv.org/abs/2103.03206) model. A videoconference record typically contains $>3000$ words and $>30$ slides, which translate to more than $3000$ textual tokens and roughly $16 * 16 * 30 = 7680$ visual tokens (with 16\*16 patches per image as in `ViT`).

In order to use a bag representation with a Transformer model on this kind of data, one should for instance use sliding windows over the input to feed the maximum token size, but the model would be losing one of its best features: the long range attention.

Graph representation allows for creative use of the structural information to construct a graph that models complex structures in order to better fit our needs.
