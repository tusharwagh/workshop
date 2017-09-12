package com.cloudera.workshop

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{SparkSession, functions}

object ProblemFour_StopwordsRemover {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("ProblemFour_StopwordsRemover")
      .getOrCreate()

    //The Data
    val data = Seq(
      (0, Seq("It", "was", "a", "bright", "cold", "day", "in", "April", "and", "the", "clocks", "were", "striking", "thirteen")),
      (1, Seq("The", "sky", "above", "the", "port", "was", "the", "color", "of", "television", "tuned", "to", "a", "dead", "channel")),
      (2, Seq("It", "was", "love", "at", "first", "sight"))
    )

    val dataSet = spark.createDataFrame(data).toDF("id", "raw")
    dataSet.show()

    /**
      * Implement the Stop word remover
      * Use the Input and output column specification
      * Print the resulting output.
      */
      val stopwords = spark.sparkContext.textFile("data/stopwords.txt").collect()
   val stopwordsRemover = new StopWordsRemover().setInputCol("raw").setOutputCol("filtered").setStopWords(stopwords).setCaseSensitive(false)
    val removerdataframe = stopwordsRemover.transform(dataSet)
removerdataframe.select("filtered").show(false)
    spark.stop()
  }
}
// scalastyle:on println
