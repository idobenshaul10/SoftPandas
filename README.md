# SoftPandas

### TODO:
1. Add saving methods for SoftDataFrame
2. Add dealing with Nans 

### Example Usage:
```commandline
import pandas as pd
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

df_filtered_image = df.soft_query("'IMAGE' ~= 'red and black swim shorts'")
relevant_price_items = df_filtered_image.query("PRICE < 600")
```