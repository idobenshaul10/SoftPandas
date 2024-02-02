import pandas as pd
from core.soft_dataframe import SoftDataFrame
from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                            metric=cosine_similarity, threshold=0.85)


lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                            metric=cosine_similarity, threshold=0.85)

df = pd.read_csv("sample_data/men-swimwear.csv")

df = SoftDataFrame(df, soft_columns=['NAME', 'DESCRIPTION & COLOR'],
                   language_model=lang_model,
                   )
#

print(df.soft_query("'DESCRIPTION & COLOR' ~= 'blue'", threshold=0.82, inplace=False))

# filtered_df = df[df.similar_to('name', 'black dress', threshold=0.85)]

# filtered_df = filtered_df_1.soft_query("name ~ 'dress'", threshold=0.85)

