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

> For example, if a graph is used to model the friendships relations of people in a social network, then the edges will be undirected as friendship is mutual; however, if the graph is used to model how people follow each other on Twitter, then the edges are directed. Depending on the edges’ directionality, a graph can be directed or undirected.

*From DGL's [graph 101](https://docs.dgl.ai/en/0.8.x/guide/graph-basic.html).*

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSVVYJclDBmKnQWPKYsw4IU2aJ4AIySSusuCw&usqp=CAU)

*An undirected graph with 6 nodes (labeled 0 to 5) and 8 (undirected) edges*

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

Even for textual documents, graph representation come with a lot of advantages. As of 2022, Bag-Of-Words representation (through the Transformer architecture) is the *de facto* standard in Natural Language Processing tasks.
Graph representation though is starting to receive a lot of interest from the NLP community: as shown in Wu *et al*'s [survey](https://arxiv.org/abs/2106.06090) on Graph Neural Networks for NLP.

For multi-modal documents, *ie* documents that contain more than one modality (such as text / image / audio), the structural information is richer than in textual documents and cannot always be represented as sequential.

#### Graphs make a lot of sense for representing such data

To illustrate this, we will construct a simple graph representation of a videoconference record, by taking advantage of the structure of the data.

We will assume that the records consist of a shared-screen video stream and an audio stream, that are being preprocessed so that we have:
* the slides as images and the time frames for which they appear on screen
* the textual transcript of the audio, with the time frames for which each sentence is being said

With only this information, we can construct this graph:

![](https://i.ibb.co/zNx9L9z/mm-graph-record.png)

Nodes are of 3 types:
* `slide nodes` are the still images from the screen-share stream
* `word nodes` are the words from the transcript
* `sentence nodes` are the sentences from the transcript

Edges are of 5 types:
* `contiguous` that represent the sequential relationships: a word and the following/previous one in a sentence
* `simultaneous` that represent relationships between audio and visual happening at the same time
* `in` that represent relationships between words and the sentence there are in
* `bag` that represent fully-connected sub-graphs; *ie* each sentence (resp. slide) is connected to all the other sentences (resp. slides)
* `self` that represent an edge from a node to itself, but is not represented in the illustration

Node (respectively edge) types are encoded in the node (resp. edge) features.

#### Graphs allows to find balance between faster inference and denser representation

A graph representation is sparser than a multimodal bag of tokens by design. It also contains more structural information than a sequence representation, but it is denser.

The graph representation allows to add or delete edges, in order to find a balance between a fully connected and denser representation. That allows the model to select the relevant nodes and understand long range dependencies, and a sparser representation, that can be processed by models with a lower computational complexity.

Since representation and model choices are very much linked, we will assume that a choice of representation comes with the associated choice of model (RNN with sequential representation, transformers with bag of tokens, and GNN with graph representation). This is consistant with the insight that transformers are a special case of GNN, as explained in [Chaitanya Joshi's blog](https://graphdeeplearning.github.io/post/transformers-are-gnns/). Similarly, one can prove that RNN are also a special case of GNN. 

Arguably, the model can be seen as the same, but with different capabilities depending on the chosen representation. And the choice of representation gives a way to balance between a faster and a more powerful model.

#### Example in favor of a sparser representation than the bag of tokens

Typically, a videoconference record is ~$40$ minutes long, the transcript is $~337$ sentences long, or ~$4,600$ words long and there are ~$46$ slides. These values were computed on average from records from the [German National Library of Science and Technology's archive](https://av.tib.eu/).

Textual input are usually tokenized using for instance a WordPiece tokenizer (such as in [BERT](https://towardsdatascience.com/how-to-build-a-wordpiece-tokenizer-for-bert-f505d97dddbb)) or a Byte-Pair Encoding (BPE) (such as in [GPT](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer)). More information about tokenizers can be found in the [Hugging Face's `transformers` documentation](https://huggingface.co/docs/transformers/tokenizer_summary).

In both cases, that means that the number of tokens is greater than the number of words. According to [Tabarak Khan's article](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) on OpenAI, a rule of thumb gives:

$$1 \text{ token} \approx \frac{3}{4} \text{ words}$$

Transformer models usually come with a size limit of a few hundred tokens, `512` tokens for BERT for instance. There are Transformer variants that are designed specifically for longer inputs: the input size limit is `4096` tokens for the [BigBird](https://arxiv.org/abs/2007.14062) model, and `2048` tokens for the [Perceiver](https://arxiv.org/abs/2103.03206) model.

For our average videoconference record, the transcript alone is estimated to be $4,600 * \frac{4}{3} \approx 6,133 $ tokens long, which already exceeds all the current transformers' token limits. 

To this we could add the visual tokens that can go up to roughly $16 * 16 * 46 = 11,776$ visual tokens (with 16\*16 patches per image as in `ViT`). However for simplicity we will assume that each slide is encoded as a single token.

In order to use a bag representation with a Transformer model on this kind of data, one should for instance use sliding windows over the input to feed the maximum token size, but the model would be losing one of its best features: the long range attention.

#### Graph representation allows for long range dependencies with manageable complexity

Let us show how the proposed graph representation compares to sequence and bag of tokens representations in term of computational complexity of the associated model as well as in term of modelisation of long range dependencies.

##### Long range dependencies modelisation

By design, our graph representation allows for long range attention over the sentences, as well as long range attention over the slides, because of the fully-connected subgraphs.
The sequence representation is not suited for long range dependency modelisation. Bag of tokens representation however is handling long range dependencies with global attention. 

#### Complexity evaluation

For the computations, we will use an example record. The video is [Firebird: The High Performance RDMS](https://av.tib.eu/media/54686), which is a record from a presentation of Patrick Fitzgerald during openSUSE Virtual Conference 2021. We extracted the transcript using OpenAI's [whisper-medium](https://huggingface.co/openai/whisper-medium) model, and the keyframes using a custom algorithm that compares the [perceptual hashes](https://github.com/JohannesBuchner/imagehash) of consecutive frames to detect the change of slides.

The records lasts `36:18`, from which we extracted $483$ sentences, $5,355$ words and $47$ slides.

##### Nodes: 
The $47$ slides amount to $47$ visual tokens, with the assomption that each slide is encoded as one token (*ie* one node).

For sequence and bag of tokens representation, we use WordPiece tokenization for words, and obtain $5,790$ text tokens.
The total amount of nodes (*ie* multimodal tokens) for these representations is $5,790 + 47 = \textbf{5,837}$.

For the graph representation, we split the document in sentences, and use word tokenization for words, each sentence will also be represented as a sentence token.
In total there are $ 5,355+483 = 5,838$ text tokens, so the total of nodes (*ie* multimodal tokens) for this representation is $5,838 + 47 = \textbf{5,885}$.

For computations, we will name

* $n$ the total number of multimodal tokens (which is different depending on the representation),

* $w$ the number of words,

* $v$ the number of slides,

* $s$ the number of sentences.

Here is the adjacency matrix for the graph representation:

![](https://i.ibb.co/sWPrPFV/adj-mat.png)

##### Edges:
In the sequence representation, each multimodal token (except from the last) is connected to the following, thus there are $\textbf{5,836}$ edges.
In the bag of tokens representation, each token is connected to every tokens, thus there are $5,837 ^2 = \textbf{34,070,569}$

For the graph representations, there are:

$\textbf{257,100}$ edges, including:

* $8,778$ `contiguous` edges ($e_{contiguous} = 2 * (w - 2 * s)) < 2 * w$),
* $2,114$ `simultaneous` edges ($2  * s \le e_{simultaneous} \le 2 * s * v$),
* $5,355$ `in` edges ($e_{in}=w$),
* $235,498$ `bag` edges ($e_{bag}=v^2 + s^2$),
* $5,355$ `self` edges ($e_{self}=w$).

We estimate the computational complexities of the models for each representation. While the sequential representation is linear in term of the document size, the bag of tokens representation is quadratic in term of the document size. For large documents, this tends to make the latter impractical to compute.

The graph representation can stand anywhere in between linear and quadratic, depending on the graph construction method.

In the example we chose, the number of edges (which is related to the complexity in our hypothesis), is smaller than $4 w + 2 s v + v^2 + s^2 = (s+v)^2 + 4w$, which gives a complexity in $O((s+v)^2 + w)$.
That means that the complexity is quadratic in the number of slides and sentences, and linear in the number of words. Since $w$ is way bigger than $s$ and $v$, this allows for faster computation than the bag of tokens representation.


In short:

|                                    | Sequence       | Proposed graph                              | Bag of tokens         |
|:------------------------------------|:----------------:|:---------------------------------------------:|:-----------------------:|
| Long range dependency modelisation | No            | Yes (over sentences and slides)             | Yes |
| Number of nodes / tokens, $n$      | $5,837$          | $5,885$                                       | $5,837$                 |
| Number of edges $e$                | $5,836$          | $257,100$                                     | $34,070,569$            |
| Complexity (compared to sequence)  | $1$              | $\approx{*44}$                                | $\approx{*5,838}$         |
| Complexity (in term of document size)          | Linear: $O(n)$ | $n = s + w + v$, $O(w + (s + v)^2)$ | Quadratic: $O(n^2)$   |

The proposed graph allows computations for $\approx 0.75\%$ of the bag of tokens computations.

This lower computation is related to the fact that the adjacency matrix for this graph is $\approx 99.25 \%$ sparse. A bag of tokens representation would be equivalent to a fully dense matrix ($0\%$ sparse), while a sequential representation would be equivalent to a diagonal matrix ($\approx 99.98\%$ sparse).

#### Limitations

For this example, we used a sequential construction of the graph. In Natural Language Processing, it is possible to create more meaningful graphs, such as [dependency graphs](https://graph4ai.github.io/graph4nlp/guide/construction/dependencygraphconstruction.html#dependency-graph-construction) for instance (one in [many possible graph constructions](https://graph4ai.github.io/graph4nlp/guide/construction.html)).

In these examples, we didn't talk about the features of the nodes, nor the features of the edges. In this example, we might want to initialize node features using common Natural Language Processing & Computer Vision methods for encoding, such as Word2Vec, sentence transformers, and image auto-encoder for instance. This adds an additionnal complexity before even feeding the Graph Neural Network.

### Conclusion

Graph representation follows naturally the structure of multimodal documents such as videoconference records. It allows for creative use of the structural information to construct a graph that models complex structures in order to better fit our needs, and manage computational complexity.
