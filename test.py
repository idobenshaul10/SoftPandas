import pandas as pd
from core.data_types import InputDataType
from core.soft_dataframe import SoftDataFrame
from embedders.clip_embedder import OpenClipEmbedder
from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                metric=cosine_similarity, threshold=0.85, device="cpu")


vision_model = OpenClipEmbedder('ViT-B-32-256', metric=cosine_similarity,
                                threshold=0.15, pretrained="datacomp_s34b_b86k")

df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': InputDataType.text,
                                     'DESCRIPTION & COLOR': InputDataType.text},
                   models={InputDataType.text: lang_model, InputDataType.image: vision_model}
                   )

df = df.add_soft_columns({'IMAGE': InputDataType.image}, inplace=False)
df = df.soft_query("'DESCRIPTION & COLOR' ~= 'swim shorts'")
df = df.soft_query("'IMAGE' ~= 'red and black'")
df = df.query("PRICE < 600")

print(df.head()['DESCRIPTION & COLOR'].values)


