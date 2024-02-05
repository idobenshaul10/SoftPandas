# SoftPandas - Pandas with Semantic Querying


<img src="soft_panda_logo.png" alt="SoftPandas" title="Logo" width="350">

[//]: # (![GitHub Repo stars]&#40;https://img.shields.io/github/stars/idobenshaul10/SoftPandas?style=social&#41;)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/ml_norms)](https://twitter.com/ml_norms)

https://github.com/idobenshaul10/SoftPandas/assets/41121256/82c467b7-701a-4cfd-9277-df7b63a66330


## Description:
SoftPandas is an initial package that allows you to work with pandas DataFrames and query them using semantic similarity.
This allows you to have conditions which are soft (e.g. all products that are similar to "red and black swim shorts"). 
Current version supports text and image data types, where if an image link is present, the image is downloaded and embedded using OpenClip.
Currently supports: 
1. Language Encoder Model: any model using SentenceTransformer
2. MultiModal Encoder Model: any model using OpenClip

Querying at the moment is only done using a text query. 

**This project is a work in progress! If you find any issues - please report them**
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
## Installation:
Python version 3.10 or later installed. Latest version from the GitHub repository:

[//]: # ()
[//]: # (```pip install softpandas```)

```pip install git+https://github.com/idobenshaul10/SoftPandas.git```


## Example Usage:
Let's say we want to get all red and black swim shorts that cost less than 600$:
We can load example data from a csv file and then query it using SoftPandas:

For full script:

```python demo.py```

Imports:
```
import pandas as pd
from softpandas.core.data_types import InputDataType
from softpandas.core.soft_dataframe import SoftDataFrame
from softpandas.embedders.clip_embedder import OpenClipEmbedder
from softpandas.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from sklearn.metrics.pairwise import cosine_similarity
```

Let's set up our encoders:
```commandline
lang_model = SentenceTransformerEmbedder('thenlper/gte-small',
                                metric=cosine_similarity, threshold=0.82, device="cpu")


vision_model = OpenClipEmbedder('ViT-B-32-256', metric=cosine_similarity,
                                threshold=0.25, pretrained="datacomp_s34b_b86k")
```
Then let's query using soft + hard queries:

```
df = pd.read_csv("sample_data/men-swimwear.csv")
df = SoftDataFrame(df, soft_columns={'NAME': InputDataType.text,
                                     'DESCRIPTION & COLOR': InputDataType.text, 
                                     'IMAGE': InputDataType.image},
                   models={InputDataType.text: lang_model, InputDataType.image: vision_model}
                   )

df = df.soft_query("'DESCRIPTION & COLOR' ~= 'swim shorts'")
df = df.soft_query("'IMAGE' ~= 'red and black'")
df = df.query("PRICE < 600")
print(df.head()['DESCRIPTION & COLOR'].values)
```



### Saving and loading:

```commandline
df.to_pickle("relevant_items.p")
df = pd.read_pickle("relevant_items.p")
```


### TODOs:
1. ~~Add saving methods for SoftDataFrame~~
2. ~~Method for adding new columns~~
3. Add dealing with Nans
   - ~~if a column is Nan, just ignore it~~
   - If value isn't there, it shouldn't pass condition - similar to normal querying
4. Add handling of multiple queries - ATM if it's more than one predicate, it'll crash.
5. Add indices instead of cosine - it's too slow 
6. Batching of initial encoding - 
   - don't do it one by one
   - ~~use device (cuda, mps, tpu, etc.)~~

   
### Long Term Goals:
1. Add automatic feature extraction from images into new columns
   - allows hard querying using visual data!
2. Add ability to soft query based on Image
3. Expand to more modalities
