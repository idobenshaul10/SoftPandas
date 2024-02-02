import pandas as pd
import pickle
from core.soft_dataframe import SoftDataFrame
from embedders.clip_embedder import OpenClipEmbedder
from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                         metric=cosine_similarity, threshold=0.82)

vision_model = OpenClipEmbedder('ViT-B-32-256', metric=cosine_similarity,
                        threshold=0.25, pretrained="datacomp_s34b_b86k")

df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': 'text',
                                     'DESCRIPTION & COLOR': 'text',
                                     'IMAGE': 'image'},
                   models={"text": lang_model, "image": vision_model}
                   )



df_filtered_image = df.soft_query("'IMAGE' ~= 'red and black swim shorts'", inplace=False)
df_filtered_desc = df.soft_query("'DESCRIPTION & COLOR' ~= 'red and black swim shorts'", inplace=False)

import pdb; pdb.set_trace()

# print(df.soft_query("'DESCRIPTION & COLOR' ~= 'blue'", threshold=0.82, inplace=False))
# filtered_df = df[df.similar_to('name', 'black dress', threshold=0.85)]

# filtered_df = filtered_df_1.soft_query("name ~ 'dress'", threshold=0.85)
