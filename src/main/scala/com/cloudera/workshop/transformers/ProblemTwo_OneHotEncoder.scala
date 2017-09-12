package com.cloudera.workshop

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

object ProblemTwo_OneHotEncoder {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
        .master("local[4]")
      .appName("ProblemTwo_OneHotEncoder")
      .getOrCreate()

    val data = Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    val df = spark.createDataFrame(data)

    val dataset = df.toDF("id","category")

    val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex").fit(dataset)

    val indexed = indexer.transform(dataset)

    val hotIndexer = new OneHotEncoder().setInputCol("categoryIndex").setOutputCol("categoryEnc").transform(indexed)
    hotIndexer.show()

    //Do the one hot encoding.
    //Hint: Build upon StringIndexer from the previous example

    spark.stop()
  }
}
// scalastyle:on println
