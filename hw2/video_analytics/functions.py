import pyspark.sql.functions as F
import pyspark.sql.types as T

import numpy as np


@F.udf(returnType=T.FloatType())
def model_score(
    views,
    likes,
    dislikes,
    comments_likes,
    comments_replies,
    w_1: float = 0.1,
    w_2: float = 10.0,
    w_3: float = -100.0,
    w_4: float = 2.0,
    w_5: float = 1.0,
):
    score = w_1 * views + w_2 * likes + w_3 * dislikes + w_4 * comments_likes + w_5 * comments_replies
    return score


@F.pandas_udf(T.FloatType(), F.PandasUDFType.GROUPED_AGG)
def median(scores) -> float:
    return np.median(scores)


@F.pandas_udf(T.ArrayType(T.StringType(), True), F.PandasUDFType.SCALAR)
def split_tags_custom_udf(tags):
    return tags.str.split('|')


