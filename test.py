import pandas as pd
from core.soft_dataframe import SoftDataFrame
from embedders.clip_embedder import OpenClipEmbedder
from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                         metric=cosine_similarity, threshold=0.85)

vision_model = OpenClipEmbedder('ViT-B-32',
                                metric=cosine_similarity, threshold=0.85)

df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': 'text',
                                     'DESCRIPTION & COLOR': 'text',
                                     'IMAGE': 'image'},
                   language_model=lang_model,
                   vision_model=vision_model
                   )
#

print(df.soft_query("'DESCRIPTION & COLOR' ~= 'blue'", threshold=0.82, inplace=False))

# filtered_df = df[df.similar_to('name', 'black dress', threshold=0.85)]

# filtered_df = filtered_df_1.soft_query("name ~ 'dress'", threshold=0.85)
