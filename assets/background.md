# Keyword Search

### Inverted Index

The index underlying the keyword search is an *inverted index*. That is, for each unique token in the corpus, it lists the documents in which each token occurs to enable fast retrieval. Unique tokens are usually determined after some preprocessing operation (e.g., lowercasing, tokenization, and/or stemming). 

### Inverted Index Search

Using the BM25F (Best Match 25 with Extension to Multiple Weighted Fields) algorithm, we search for the most relevant documents for the given query. Essentially, BM25 is an extension of the TF-IDF measure (the "F" indicates that we can search across different fields and weight them differently). It calculates a score for each document, which increases roughly in
- the term frequency (`tf`) of the query terms in the document,
- the inverse document frequency (`idf`) of the query terms in the document,
- the saturation parameter `k_1`, which determines how repeated mentions of a query term in the document should contribute to the relevance score (higher leads to slower saturation), and
- the bound parameter `b`, which favors matches in documents that are short relative to the average document in the corpus.

# Vector Search

### Vector Index

The index underlying the vector search is based on Hierarchical Navigable Small World (HNSW) graphs. HNSW graphs consist of a couple of layers that arrange vectorized documents in a hierarchical manner (*sub-graphs*). The top layer (layer `N`) contains a limited set of vectors, called *entry points*. From layer `N-1` to layer `0` we successively increase the number of vectors, so that layer `0` contains the full graph. Thus, vectors on upper layers have fewer edges (i.e., connections with neighboring vectors) compared to vectors on lower layers. The number of edges per node is defined by the `maxConnections` parameter.

### Vector Index Search

Using an approximate nearest neighbor (ANN) algorithm, we search for the nearest neighbors of a given query vector by traversing through the HNSW graph:
1. Select the entry point on layer `N` and pick the neighbor closest to the query vector.
2. Proceed to layer `N-1`. Starting from the previously chosen neighbor on layer `N`, choose the nearest neihbor with the smallest distance to the query vector.
3. Continue until layer `0` is reached by greedily selecting the nearest neihbor that is closest to the query vector on each iteration.

By specifying the `ef` parameter, we can ensure that at each step, a dynamic list of `ef` nearest neighbor (instead of a single neihbor) is retained and evaluated at the next layer, which reduces performance but potentially increases the accuracy of the vector search.

To evaluate "closeness" between vectors, we rely on one of several possible *distance metrics*.
- *Dot product similarity*: Does not account for the vector magnitudes and, thus may be a reasonable choice if we want to emphasize the strength of the embedding signals. Since signals (i.e., numerical values in any of the `d` vector dimensions) tend to be larger for longer documents, dot product comparisons favor longer documents.
- *Cosine similarity*: Normalizes vectors and can therefore be a reasonable choice in semantic search where we are primarily interested in the directions (i.e., contents) of the vectors not the magnitudes. Due to the normalization, each vector contributes equally to the distance calculation.

### Vectorization Module

Documents are vectorized using a *vectorization module*. One popular option for representing text as vectors would be a text embedding model (e.g., OpenAI's GPT models, sentence transformers, or Cohere embedding models). Ideally, the chosen embedding model should be pretrained on data that reflects
- the language of the given corpus,
- the document domain (i.e., the genre and subject matter of the corpus),
- the target search task, e.g., 
    - exact semantic search (finding documents with similar semantic content)
    - relational semantic search (finding documents with a certain relationship to each other, e.g., entailment or Q&A),
- the average document length of the given corpus,
and weighs accuracy and performance (captured by the dimensionality `d` of the embedding space).

Embedding models which serve to vectorize and search over documents are typically *bi-encoder* models. These encode a single doc into a fixed length vector and are trained using a contrastive learning objective. They allow to efficiently compare two docs by embedding each one and computing a distance metric (e.g., cosine similarity).

### Reranking Module

Search results can be re-ranked using the *reranker module*. The idea is to the re-order search results by relevance using a more computationally intensive transformer model that specifically assesses the relevance of a given search result to the initial query (note: reranking can also be performed based on a query that is different from the initial search query).

Embedding models which serve to rerank search results are typically *cross-encoder* models. These encode two docs simultaneously, capturing relationships between both docs. Since they are pretrained on a classification objective, they output a single score indicating the relatedness between the two documents (i.e., signaling the relevance of the search result to the query). They are much slower (because they combinatorially encode pairs of docs at the same time), but more accurate than *bi-encoder* models, and thus allow re-ranking of a limited set of search results.


### Quantization Module

To reduce the size of the embedded corpus on disk, we could employ compression techniques to coarsen the embedded vectors (ideally without incurring a notable degradation in expressiveness). One popular compression technique for vectors is refered to as *product quantization* and works as follows:
1. Partition each vector of dimensionality `d` into `n_seg` segments, i.e., *sub-vectors*. Each segment then comprises of `d/n_seg` embedding dimensions.
2. For each of the `n_seg` run *k-means clustering* (or an alternative clustering algorithm) to compute cluster centroids, such that each and every segment can be represented by a finite set of centroids. Each centroid can later be identified by a unique centroid index (e.g., `s2_c4` for the 4th centroid in segment 2), so that any `d`-dimensional vector can be represented by `n_seg` centroid indices.

Note that the k-means model can be fitted using a fraction of the total vector database. For example, we can start quantization (i.e., fitting the various k-means models) after x% of our total corpus has been embedded. This reduces the computational complexity of fitting the k-means models, and is appropriate insofar as the already embedded documents reflect the entire corpus (i.e., capture the true data distribution).

While vector search using quantized vectors improves performance, the coarsened vectors can cause a degradation in recall. As a remedy, we could increase `maxConnections` to ensure that the relevant vectors end up in the search list on each layer of the HNSW graph.

# Hybrid Search

Hybrid search combines keyword and vector search by aggregating/weighing the different relevance scores. At the heart, hybrid search performs two searches, and based on either the rankings or the relevance scores of the search results, weights (i.e., fuses) both results to produce an aggregate ranking that is plausibly more refined compared to performing any of the two searches separately. `weaviate` provides two approaches to fusing search results:
1. *Ranked fusion*, which weights the raw rank of each document in the retrieved output (e.g., 50%:50%)
2. *Relative score fusion*, which normalizes the relevance scores to the [0;1] interval before weighing the two outputs
