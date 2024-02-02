import pandas as pd
from core.soft_dataframe import SoftDataFrame
from embedders.embedder import SentenceTransformerEmbedder

from sklearn.metrics.pairwise import cosine_similarity

lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                            metric=cosine_similarity, threshold=0.85)
df = SoftDataFrame(pd.read_csv("sample_data/Womens Fashion.csv"),
                   soft_columns=['name'], language_model=lang_model)
#
# df.soft_query("name ~= 'black dress'", threshold=0.85, inplace=True)
filtered_df = df[df.similar_to('name', 'black dress', threshold=0.85)]
import pdb; pdb.set_trace()
# filtered_df = filtered_df_1.soft_query("name ~ 'dress'", threshold=0.85)

