"""Construction of Weaviate graph database from transcripts."""
import json
from pathlib import Path

import pandas as pd
import weaviate
import weaviate.classes as wvc
from tqdm import tqdm


with Path('assets/config.json').open(encoding='utf-8') as f:
    PATH_DATA = Path(json.load(f)['data-sentences'])
    PATH_META = Path(json.load(f)['data-meta'])
    IMPORT_BATCH_SIZE = json.load(f)['import-batch-size']
    DB_SIZE = json.load(f)['database-size']


client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    timeout=(5, 15),
)

if __name__ == "__main__":

    # init collection
    if client.collections.exists('Sentences'):
        client.collections.delete('Sentences')

    # define document collection
    client.collections.create(
        name='Sentences',
        properties=[
            wvc.Property(
                name='doc_id',
                data_type=wvc.DataType.TEXT,
                indexFilterable=False, indexSearchable=False,
                skip_vectorization=True,
            ),
            wvc.Property(
                name='sa_id',
                data_type=wvc.DataType.INT,
                indexFilterable=True, indexSearchable=False,
                skip_vectorization=True,
            ),
            wvc.Property(
                name='title',
                data_type=wvc.DataType.TEXT,
                indexFilterable=True, indexSearchable=True,
                skip_vectorization=True,
                tokenization=wvc.Tokenization.LOWERCASE,
            ),
            wvc.Property(
                name='coname',
                data_type=wvc.DataType.TEXT,
                indexFilterable=True, indexSearchable=True,
                skip_vectorization=True,
                tokenization=wvc.Tokenization.LOWERCASE,
            ),
            wvc.Property(
                name='fy',
                data_type=wvc.DataType.INT,
                indexFilterable=True, indexSearchable=False,
                skip_vectorization=True,
            ),
            wvc.Property(
                name='q',
                data_type=wvc.DataType.INT,
                indexFilterable=True, indexSearchable=False,
                skip_vectorization=True,
            ),
            wvc.Property(
                name='section',
                data_type=wvc.DataType.TEXT,
                indexFilterable=True, indexSearchable=False,
                skip_vectorization=True,
                tokenization=wvc.Tokenization.LOWERCASE,
            ),
            wvc.Property(
                name='speaker',
                data_type=wvc.DataType.TEXT,
                indexFilterable=True, indexSearchable=True,
                skip_vectorization=True,
                tokenization=wvc.Tokenization.LOWERCASE,
            ),
            wvc.Property(
                name='role',
                data_type=wvc.DataType.TEXT,
                indexFilterable=True, indexSearchable=False,
                skip_vectorization=True,
                tokenization=wvc.Tokenization.WORD,
            ),
            wvc.Property(
                name='text',
                data_type=wvc.DataType.TEXT,
                vectorize_property_name=False,
                tokenization=wvc.config.Tokenization.WORD,
                #quantizer=wvc.Reconfigure.VectorIndex.Quantizer.pq(),
            )
        ],
        inverted_index_config=wvc.Configure.inverted_index(),
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(
            pooling_strategy='masked_mean',
            vectorize_collection_name=False,
        ),
        vector_index_config=wvc.Configure.VectorIndex.hnsw(
            distance_metric=wvc.VectorDistance.COSINE,
            ef_construction=512,
            max_connections=64,
            ef=512,
        ),
        generative_config=None,
    )

    # import data and attach metadata
    data = pd.read_feather(PATH_DATA).head(n=DB_SIZE)
    meta = pd.read_pickle(PATH_META)[['sa_id', 'title', 'coname', 'fy', 'q']]
    meta[['fy', 'q']] = meta[['fy', 'q']].astype(int)
    data = data.merge(meta, on='sa_id', how='left')
    data = data[data['fy'].notna() & data['q'].notna()]

    db = client.collections.get('Sentences')

    COUNTER = 0
    pbar = tqdm(total=len(data))
    while COUNTER < len(data):
        blobs = []
        for idx, row in data.iloc[COUNTER:COUNTER+IMPORT_BATCH_SIZE].iterrows():
            blobs.append({
                'doc_id': f"{row['sa_id']}_{row['remark_id']}_{row['sent_id']}",
                'sa_id': row['sa_id'],
                'title': row['title'],
                'coname': row['coname'],
                'fy': row['fy'],
                'q': row['q'],
                'section': row['section'],
                'speaker': row['speaker'],
                'role': row['role'],
                'text': row['text'],
            })
        db.data.insert_many(blobs)
        COUNTER += IMPORT_BATCH_SIZE
        pbar.update(IMPORT_BATCH_SIZE)
    pbar.close()

    # check imports
    n = db.aggregate.over_all(total_count=True).total_count
    d = len(db.query.fetch_objects(limit=1, include_vector=True).objects[0].vector)
    print("\nConstruction successful. Size of graph database on disk:\n",
          f"- Number of embedded docs: {n}\n",
          f"- Size of vectors on disk: {n} x {d} x 4 = {n*d*4} bytes = {round(n*d*4/1e9, 2)} GB\n",
          f"- Size of graph on disk: {n} x 64 x 8 = {n*64*8} bytes = {round(n*64*8/1e9, 2)} GB\n")
