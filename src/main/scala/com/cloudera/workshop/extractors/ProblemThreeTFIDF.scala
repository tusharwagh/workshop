package com.cloudera.workshop.extractors

import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.SparkSession

object ProblemThreeTFIDF {

def main(args: Array[String]) {

    val spark = SparkSession
      .builder
        .master("local[4]")
      .appName("ProblemThreeTFIDF")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")

    val tokens = new RegexTokenizer()
    .setGaps(false)
    .setPattern("\\w+")
    .setInputCol("sentence")
    .setOutputCol("words")
    .transform(sentenceData)
  println(tokens.collectAsList())
  //tokens.show(4)

  /*val filteredTokens = new StopWordsRemover()
    .setInputCol("bookWords")
    .setOutputCol("filteredwords")
    .setStopWords(spark.sparkContext.textFile(stopWordFile).collect())
    .transform(tokens)*/
  //filteredTokens.show(7)

  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawfeatures").setNumFeatures(50);
  val featurizedData = hashingTF.transform(tokens)

  //println(featurizedData.select("filteredwords").first().size)
  //featurizedData.select("filteredwords","rawfeatures").show(1,false)

  val idf = new IDF().setInputCol("rawfeatures").setOutputCol("features").fit(featurizedData).transform(featurizedData)
  idf.select("words","rawfeatures","features").show(false)

   spark.stop()
  }
}
