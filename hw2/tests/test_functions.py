import pytest
import pyspark.sql.functions as F
import pyspark.sql.types as T
from chispa import *
from pyspark.sql import SparkSession

from video_analytics.functions import (
    split_tags_custom_udf,
    median,
    model_score
)


@pytest.fixture(scope='session')
def spark():
    return (
        SparkSession
        .builder
        .master("local")
        .appName("chispa")
        .getOrCreate()
    )


def test_median(spark):
    data = [
        ("a", 1.0),
        ("a", 2.0),
        ("a", 3.0),
        ("a", 4.0),
        ("a", 5.0),
        ("b", None),
        ("c", 1.0),
        ("c", 2.0),
        (None, None)
    ]
    median_scores = [
        ("a", 3.0),
        ("b", None),
        ("c", 1.5),
        (None, None)
    ]

    median_scores_df = (
        spark
        .createDataFrame(
            median_scores,
            schema=T.StructType([
                T.StructField("class", T.StringType(), True),
                T.StructField("median_score", T.FloatType(), True),
            ])
        )
    )
    df = (
        spark
        .createDataFrame(
            data,
            schema=T.StructType([
                T.StructField("class", T.StringType(), True),
                T.StructField("score", T.FloatType(), True),
            ])
        )
        .groupBy("class")
        .agg(
            median(F.col('score')).alias('median_score')
        )
    )

    assert_df_equality(df, median_scores_df, ignore_row_order=True)


def test_split_tags_custom_udf(spark):
    data = [
        ("cat|dog", ["cat", "dog"]),
        ("", [""]),
        ("cat", ["cat"]),
        (None, None)
    ]
    df = (
        spark
        .createDataFrame(data, ["tags", "expected_tags"])
        .withColumn("parsed_tags", split_tags_custom_udf(F.col("tags")))
    )

    assert_column_equality(df, "parsed_tags", "expected_tags")


def test_model_score(spark):
    data = [
        (1., 1., 1., 1., 1., 0., 5.),
        (0., 0., 0., 0., 0., 0., 0.),
    ]
    df = (
        spark
        .createDataFrame(
            data,
            schema=T.StructType([
                T.StructField("views", T.FloatType(), True),
                T.StructField("likes", T.FloatType(), True),
                T.StructField("dislikes", T.FloatType(), True),
                T.StructField("comments_likes", T.FloatType(), True),
                T.StructField("comments_replies", T.FloatType(), True),
                T.StructField("expected_zero_weight", T.FloatType(), True),
                T.StructField("expected_one_weight", T.FloatType(), True),
            ])
        )
        .withColumn(
            'zero_weight',
            model_score(
                F.col('views'),
                F.col('likes'),
                F.col('dislikes'),
                F.col('comments_likes'),
                F.col('comments_replies'),
                F.lit(0.), F.lit(0.), F.lit(0.), F.lit(0.), F.lit(0.),
            )
        )
        .withColumn(
            'one_weight',
            model_score(
                F.col('views'),
                F.col('likes'),
                F.col('dislikes'),
                F.col('comments_likes'),
                F.col('comments_replies'),
                F.lit(1.), F.lit(1.), F.lit(1.), F.lit(1.), F.lit(1.),
            )
        )
    )

    assert_column_equality(df, "zero_weight", "expected_zero_weight")
    assert_column_equality(df, "one_weight", "expected_one_weight")

