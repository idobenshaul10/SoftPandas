import pandas as pd
import softpandas as sp
from softpandas.core.data_types import InputDataType
from softpandas.core.soft_dataframe import SoftDataFrame
from softpandas.embedders.clip_embedder import OpenClipEmbedder
from softpandas.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                metric=cosine_similarity, threshold=0.82, device="cpu")

vision_model = OpenClipEmbedder('ViT-B-32-256', metric=cosine_similarity,
                                threshold=0.15, pretrained="datacomp_s34b_b86k")

df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': InputDataType.text,
                                     'DESCRIPTION & COLOR': InputDataType.text},
                   models={InputDataType.text: lang_model, InputDataType.image: vision_model},
                   num_voronoi_clusters=5
                   )

# df = df.soft_query("'DESCRIPTION & COLOR' ~= 'red and black'", threshold=0.8)
# df = df.add_soft_columns({'IMAGE': InputDataType.image}, inplace=False)

df.soft_query("'DESCRIPTION & COLOR' ~= 'swim shorts'", threshold=0.85)
import pdb; pdb.set_trace()
df = df.soft_query("'DESCRIPTION & COLOR' ~= 'swim shorts'", threshold=0.85)

df.soft_query("'IMAGE' ~= 'swim shorts'", threshold=0.15)


df = df.soft_query("'IMAGE' ~= 'swim shorts'")

df = df.query("PRICE < 600")

print(df.head()['DESCRIPTION & COLOR'].values)


