package com.cloudera.workshop.transformers

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object ProblemOne_IndexToString {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("ProblemOne_IndexToString")
      .getOrCreate()

    //Given the following data set, create a dataset with columns labelled as "id" and "category"\

    val data = Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    //Step 1:  Create a DataFrame
    val df = spark.createDataFrame(data).toDF("id","category")

    //Step 2: Transform to a Dataset
    val mydata = df.toDF("id","category")

    //Step 3: Initialize a StringIndexer to fit the DataSet
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(mydata);
    val indexed = indexer.transform(mydata)

    println(s"Transformed string column '${indexer.getInputCol}' " +
      s"to indexed column '${indexer.getOutputCol}'")
    indexed.show()

    val inputColSchema = indexed.schema(indexer.getOutputCol)
    println(s"StringIndexer will store labels in output column metadata: " +
      s"${Attribute.fromStructField(inputColSchema).toString}\n")
    //Step 4: Convert it back to the original String.
    val convert = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory").transform(indexed)
    convert.show()
    spark.stop()
  }
}
// scalastyle:on println
