# SoftPandas

### TODO:
1. ~~Add saving methods for SoftDataFrame~~
2. Method for adding new columns
3. Add dealing with Nans 
4. Batching of initial encoding - 
   - don't do it one by one
   - use device (cuda, mps, tpu, etc.)
5. make into a package
   - requirements file 
   

### Example Usage:
1. Let's say we want to get all red and black swim shorts that cost less than 600$
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

df = df.soft_query("'IMAGE' ~= 'red and black swim shorts'")
relevant_items = df.query("PRICE < 600")
```
2. Saving and loading:
```commandline
relevant_price_items.to_pickle("relevant_items.p")
a = pd.read_pickle("relevant_items.p")
```