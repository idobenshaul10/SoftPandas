from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

tqdm.pandas()


class SoftDataFrame(pd.DataFrame):
    _metadata = ['soft_columns', 'models', 'hidden_columns']

    # Constructor properties
    @property
    def _constructor(self):
        return SoftDataFrame

    @property
    def _constructor_sliced(self):
        # This returns a pd.Series by default. If you've created a custom Series class, return that instead.
        return pd.Series

    @property
    def _constructor_expanddim(self):
        # This is less commonly used but should return your custom class for operations
        # that change the dimensionality of the data, for example pd.DataFrame.pivot
        return SoftDataFrame

    def __init__(self, *args, soft_columns: List[Any] | Dict[Any, Any] = None,
                 models: Dict[str, Any] = None, reembed=True, **kwargs):

        super().__init__(*args, **kwargs)
        self.models = models if models is not None else {}
        self.hidden_columns = []

        if soft_columns:
            if isinstance(soft_columns, list):
                self.soft_columns = {k: "text" for k in soft_columns}
            else:
                self.soft_columns = soft_columns
        else:
            self.soft_columns = {}

        if reembed:
            self.embed_soft_columns()

    def __finalize__(self, other, method=None, **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self


    def embed_soft_columns(self):
        for col, data_type in self.soft_columns.items():
            new_column_name = f"{col}_{data_type}_embeddings"
            if new_column_name in self.columns:
                continue
            if data_type in self.models:
                self[new_column_name] = self[col].progress_apply(self.models[data_type].encode)
            else:
                raise ValueError(f"Model for data type '{data_type}' not found.")
            self.hidden_columns.append(new_column_name)

    def similar_to(self, col: str, value: str, **kwargs):
        # TODO: Add support for nan values
        data_type = self.soft_columns[col]
        if data_type in self.models:
            semantic_model = self.models[data_type]
            query_embedding = semantic_model.encode(value)
        else:
            raise ValueError(f"Model for data type '{data_type}' not found.")

        column_embeddings = np.stack(self[f"{col}_{data_type}_embeddings"])
        similarity_scores = semantic_model.metric([query_embedding], column_embeddings).flatten()
        threshold = kwargs.get('threshold', semantic_model.threshold)
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
                                   models=self.models,
                                   reembed=False)
            return result

    # def to_pickle(self, path: str) -> None:
    #     pickle.dump(self, open(path, "rb"))
