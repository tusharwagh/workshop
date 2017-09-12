package com.cloudera.workshop

import org.apache.spark.ml.feature.{NGram, RegexTokenizer, Tokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

object ProblemThree_Tokenizer {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
        .master("local[4]")
      .appName("ProblemThree_Tokenizer")
      .getOrCreate()

    val data = Seq(
      (0, " It was a bright cold day in April, and the clocks were striking thirteen."),
      (1, "The sky above the port was the color of television, tuned to a dead channel."),
      (2, "It was love at first sight.")
    )

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens = udf { (words: Seq[String]) => words.length }

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
        .withColumn("tokens", countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
        .withColumn("tokens", countTokens(col("words"))).show(false)

    val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    ngram.transform(tokenized).show()

    spark.stop()
  }
}
// scalastyle:on println
