package com.cloudera.workshop.extractors

import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession

object ProblemBooksTFIDF {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("ProblemFour_StopwordsRemover")
      .getOrCreate()

    val textRDD = spark.sparkContext.wholeTextFiles("data/books/all")
    val booksTFIDF = spark.createDataFrame(textRDD).toDF("bookName","bookContent")
    //booksTFIDF.show(4)

    var stopWordFile = "data/stopwords.txt"

    val tokens = new RegexTokenizer()
      .setGaps(false)
      .setPattern("\\w+")
      .setMinTokenLength(4)
      .setInputCol("bookContent")
      .setOutputCol("bookWords")
      .transform(booksTFIDF)

    //tokens.show(4)

    val filteredTokens = new StopWordsRemover()
      .setInputCol("bookWords")
      .setOutputCol("filteredwords")
      .setStopWords(spark.sparkContext.textFile(stopWordFile).collect())
      .transform(tokens)
    //filteredTokens.show(7)

    val hashingTF = new HashingTF().setInputCol("filteredwords").setOutputCol("rawfeatures").setNumFeatures(230);
    val featurizedData = hashingTF.transform(filteredTokens)

    println(featurizedData.select("filteredwords").first().size)
    //featurizedData.select("filteredwords","rawfeatures").show(1,false)

    val idf = new IDF().setInputCol("rawfeatures").setOutputCol("features").fit(featurizedData).transform(featurizedData)
    idf.select("filteredwords","rawfeatures","features").show(5,false)
    spark.close()

  }

}
