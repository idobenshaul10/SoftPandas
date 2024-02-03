from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import pickle

from core.data_types import InputDataType

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

    def __init__(self, *args, soft_columns: List[Any] | Dict[Any, InputDataType] = None,
                 models: Dict[InputDataType, Any] = None, reembed=True, **kwargs):

        super().__init__(*args, **kwargs)
        self.models = models if models is not None else {}
        self.hidden_columns = set()

        if soft_columns:
            if isinstance(soft_columns, list):
                self.soft_columns = {k: InputDataType.text for k in soft_columns}
            else:
                self.soft_columns = soft_columns
        else:
            self.soft_columns = {}

        if reembed:
            self.embed_soft_columns_init()

    def __finalize__(self, other, method=None, **kwargs):
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def embed_soft_columns_init(self) -> None:
        self.add_soft_columns(self.soft_columns, inplace=True)

    def embed_soft_column(self, data_type: InputDataType, col: str, new_column_name: str) -> None:
        if col not in self.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if data_type in self.models:
            self[new_column_name] = self[col].progress_apply(self.models[data_type].encode)
        else:
            raise ValueError(f"Model for data type '{data_type}' not found.")
        # self.hidden_columns.add(new_column_name)

    def add_soft_columns(self, new_columns: Dict[str, InputDataType], inplace: bool = True) -> SoftDataFrame | None:
        for col, data_type in new_columns.items():
            new_column_name = f"{col}_{data_type.name}_embeddings"
            # if col in self.soft_columns and new_column_name in self.columns:
            #     warnings.warn(f"Semantic column for '{col}' already exists: {self.soft_columns[col]}, skipping column.")
            #     continue
            semantic_col_exists = False
            for other_data_type in InputDataType._member_names_:
                check_column_name = f"{col}_{other_data_type}_embeddings"
                if check_column_name in self.columns:
                    semantic_col_exists = True
                    break
            if semantic_col_exists:
                warnings.warn(f"Semantic column for '{col}' already exists: {self.soft_columns[col]}, skipping column.")
                continue
            self.embed_soft_column(data_type, col, new_column_name)
            self.soft_columns[col] = data_type
        if not inplace:
            return self

    def similar_to(self, col: str, value: str, **kwargs) -> np.ndarray:
        # TODO: Add support for nan values
        data_type = self.soft_columns[col]
        if data_type in self.models:
            semantic_model = self.models[data_type]
            query_embedding = semantic_model.encode(value)
        else:
            raise ValueError(f"Model for data type '{data_type}' not found.")

        column_embeddings = np.stack(self[f"{col}_{data_type.name}_embeddings"])
        similarity_scores = semantic_model.metric([query_embedding], column_embeddings).flatten()
        threshold = kwargs.get('threshold', semantic_model.threshold)
        mask = similarity_scores >= threshold
        return mask

    def soft_query(self, expr: str, inplace: bool = False, **kwargs) -> SoftDataFrame | None:
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
            return self
            # result = SoftDataFrame(filtered_data,
            #                        soft_columns=self.soft_columns,
            #                        models=self.models,
            #                        reembed=False)
            # return result

