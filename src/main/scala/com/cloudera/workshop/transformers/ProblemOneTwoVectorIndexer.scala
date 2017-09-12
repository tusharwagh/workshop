package com.cloudera.workshop

import breeze.linalg.all
import org.apache.spark.ml.feature.{Normalizer, VectorIndexer}
import org.apache.spark.sql.SparkSession

object ProblemOneTwoVectorIndexer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
        .master("local[4]")
      .appName("ProblemOneTwoVectorIndexer")
      .getOrCreate()

    //Either use the previous data or the libsvm data here

    //Init and use a Vector Indexer which combines StringIndexer and OneHotEncoder.

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    data.show()
    val indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(20)

    val indexerModel = indexer.fit(data)
    println("numfeatures "+indexerModel.categoryMaps.keys.toList)
    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " +
      categoricalFeatures.mkString(", "))

    // Create new column "indexed" with categorical values transformed to indices
    val indexedData = indexerModel.transform(data)
    indexedData.show(false)

    val normalizedData = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0)
    normalizedData.transform(indexedData).show()
    spark.stop()
  }
}
// scalastyle:on println
