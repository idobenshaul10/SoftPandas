import pandas as pd
from embedders.embedder import SentenceTransformerEmbedder
from tqdm import tqdm
import numpy as np

tqdm.pandas()


class SoftDataFrame(pd.DataFrame):
    _metadata = ['soft_columns', 'language_model', 'vision_model']

    def __init__(self, *args, soft_columns=None, language_model=None, vision_model=None,
                 reembed=True, **kwargs):

        super().__init__(*args, **kwargs)
        if soft_columns:
            self.soft_columns = soft_columns
        self.language_model = language_model
        self.vision_model = vision_model
        if reembed:
            self.embed_soft_columns()

    def embed_soft_columns(self):
        for col in self.soft_columns:
            new_column_name = f"{col}_lang_embeddings"
            if new_column_name in self.columns:
                continue
            self[new_column_name] = self[col].progress_apply(self.language_model.encode)

    def similar_to(self, col: str, value: str, **kwargs):
        query_embedding = self.language_model.encode(value)
        column_embeddings = np.stack(self[f"{col}_lang_embeddings"])
        similarity_scores = self.language_model.metric([query_embedding], column_embeddings).flatten()
        threshold = kwargs.get('threshold', self.language_model.threshold)
        mask = similarity_scores >= threshold
        return mask

    def soft_query(self, expr: str, inplace: bool = False, **kwargs):
        # Check for the presence of "~" for semantic similarity queries
        if '~=' not in expr:
            raise ValueError("Soft query must contain '~=' for semantic similarity.")
        else:
            col, value = expr.split('~=')
            col = col.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes and whitespace

            if col not in self.soft_columns:
                raise ValueError(f"Semantic similarity query not supported for column '{col}'.")
                # TODO: Add support for adding this to soft_columns

            mask = self.similar_to(col=col, value=value, **kwargs)
            filtered_data = self[mask]

        if inplace:
            self._update_inplace(filtered_data)
            return None
        else:
            result = SoftDataFrame(filtered_data,
                                   soft_columns=self.soft_columns,
                                   language_model=self.language_model,
                                   vision_model=self.vision_model,
                                   reembed=False)
            return result
