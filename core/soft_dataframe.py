import pandas as pd
from embedders.embedder import SentenceTransformerEmbedder
from tqdm import tqdm
import numpy as np
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
            self[f"{col}_lang_embeddings"] = self[col].progress_apply(self.language_model.encode)

    def soft_query(self, expr: str, inplace: bool = False, **kwargs):
        # Check for the presence of "~" for semantic similarity queries
        if '~' not in expr:
            raise ValueError("Soft query must contain '~' for semantic similarity.")
        else:
            col, value = expr.split('~')
            col = col.strip()
            value = value.strip().strip('"').strip("'")  # Remove quotes and whitespace

            if col not in self.soft_columns:
                raise ValueError(f"Semantic similarity query not supported for column '{col}'.")
                # TODO: Add support for adding this to soft_columns

            # Compute the semantic similarity
            query_embedding = self.language_model.encode(value)
            column_embeddings = np.stack(self[f"{col}_lang_embeddings"])
            similarity_scores = self.language_model.metric([query_embedding], column_embeddings).flatten()

            # Apply the threshold for similarity (this could be parameterized)
            threshold = kwargs.get('threshold', self.language_model.threshold)
            mask = similarity_scores >= threshold

            # Use the boolean mask to filter the dataframe
            filtered_df = self[mask]

        if inplace:
            self._update_inplace(filtered_df)
            return None
        else:
            return filtered_df

