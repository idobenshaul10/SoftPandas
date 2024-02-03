# SoftPandas

### TODO:
1. ~~Add saving methods for SoftDataFrame~~
2. ~~Method for adding new columns~~
3. Add dealing with Nans 
4. Batching of initial encoding - 
   - don't do it one by one
   - use device (cuda, mps, tpu, etc.)
5. make into a package
   - requirements file 
   
### Long Term Goals:
1. Add automatic feature extraction from images into new columns
   - allows hard querying using visual data!
2. Add ability to soft query based on Image

### Example Usage:
1. Let's say we want to get all red and black swim shorts that cost less than 600$
```commandline
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
df_filtered_desc = relevant_price_items.soft_query("'DESCRIPTION & COLOR' ~= 'red and black swim shorts'")
```
2. Saving and loading:
```commandline
relevant_price_items.to_pickle("relevant_items.p")
a = pd.read_pickle("relevant_items.p")
```