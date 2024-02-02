from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas()


class SoftDataFrame(pd.DataFrame):
    _metadata = ['soft_columns', 'language_model', 'vision_model']

    def __init__(self, *args, soft_columns: List[Any] | Dict[Any, Any] = None,
                 language_model=None, vision_model=None,
                 reembed=True, **kwargs):

        super().__init__(*args, **kwargs)
        if soft_columns:
            if isinstance(soft_columns, list):
                self.soft_columns = {k: "text" for k in soft_columns}
            self.soft_columns = soft_columns
        self.language_model = language_model
        self.vision_model = vision_model
        if reembed:
            self.embed_soft_columns()

    def embed_soft_columns(self):
        for col, data_type in self.soft_columns.items():
            new_column_name = f"{col}_{data_type}_embeddings"
            if new_column_name in self.columns:
                continue
            if data_type == "text":
                self[new_column_name] = self[col].progress_apply(self.language_model.encode)
            elif data_type == "image":
                self[new_column_name] = self[col].progress_apply(self.vision_model.encode)
            else:
                raise ValueError(f"Data type '{data_type}' not supported for soft columns.")

    def similar_to(self, col: str, value: str, **kwargs):
        # TODO: Add support for nan values
        query_embedding = self.language_model.encode(value)
        column_embeddings = np.stack(self[f"{col}_text_embeddings"])
        similarity_scores = self.language_model.metric([query_embedding], column_embeddings).flatten()
        threshold = kwargs.get('threshold', self.language_model.threshold)
        mask = similarity_scores >= threshold
        return mask

    def soft_query(self, expr: str, inplace: bool = False, **kwargs):
        # Check for the presence of "~=" for semantic similarity queries
        if '~=' not in expr:
            raise ValueError("Soft query must contain '~=' for semantic similarity.")
        else:
            col, value = expr.split('~=')
            col = col.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")

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
