import pandas as pd
from embedders.embedder import SentenceTransformerEmbedder
from tqdm import tqdm
tqdm.pandas()

class SoftDataFrame(pd.DataFrame):
    def __init__(self, *args, soft_columns=None, language_model=None, vision_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_columns = soft_columns if soft_columns is not None else []
        self.language_model = language_model
        self.vision_model = vision_model
        self.embed_soft_columns()

    def embed_soft_columns(self):
        for col in self.soft_columns:
            import pdb; pdb.set_trace()
            self[f"{col}_lang_embeddings"] = self[col].progress_apply(self.language_model.encode)









