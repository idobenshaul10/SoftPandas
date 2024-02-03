import pandas as pd

from core.data_types import InputDataType
from core.soft_dataframe import SoftDataFrame
from embedders.clip_embedder import OpenClipEmbedder
from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                metric=cosine_similarity, threshold=0.82, device="cpu")


vision_model = OpenClipEmbedder('ViT-B-32-256', metric=cosine_similarity,
                                threshold=0.25, pretrained="datacomp_s34b_b86k")

df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': InputDataType.text,
                                     'DESCRIPTION & COLOR': InputDataType.text,
                                     'FABRIC': InputDataType.text},
                   models={InputDataType.text: lang_model, InputDataType.image: vision_model}
                   )

relevant_price_items = df.query("PRICE < 600")
df_filtered_desc = relevant_price_items.soft_query("'DESCRIPTION & COLOR' ~= 'red and black swim shorts'")
df = df_filtered_desc.add_soft_columns({'IMAGE': InputDataType.image}, inplace=False)

df_filtered_image = df.soft_query("'IMAGE' ~= 'red and black swim shorts'")
print(df_filtered_image)
import pdb; pdb.set_trace()
