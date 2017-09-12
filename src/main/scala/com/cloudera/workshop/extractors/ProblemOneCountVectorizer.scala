package com.cloudera.workshop.extractors

import org.apache.spark.ml.feature.{CountVectorizer, IDF, RegexTokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession

object ProblemOneCountVectorizer {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
        .master("local[4]")
      .appName("ProblemOneCountVectorizer")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (1.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (2.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (3.0, "It was love at first sight.")
    )).toDF("label", "sentence")


    val tokens = new RegexTokenizer()
      .setGaps(false)
      .setPattern("\\w+")
      .setInputCol("sentence")
      .setOutputCol("words")
      .transform(sentenceData)
    val countVectorizer = new CountVectorizer()
    val countvectorsmodel = countVectorizer.setInputCol("words").setOutputCol("vectors").fit(tokens)
    //countvectorsmodel.vocabulary.foreach(x => println(x))

    val countvectors = countvectorsmodel.transform(tokens)
    //countvectors.show(false)
    countvectors
    val idf = new IDF().setInputCol("vectors").setOutputCol("features").fit(countvectors).transform(countvectors)
    //idf.select("words","vectors","features").show(false)
    //println(idf.select("features").rdd.map(_.getAs[SparseVector](i=0).indices).take(10))
    var map:Map[String,Int] = Map()
    idf.select("label","words","features").rdd.take(3).foreach(f => {
 /*     println(f.get(0))
      println(f.get(1))
      println(f.get(2))*/
      val tempRow = f.getAs[SparseVector](2).indices
      val vocalList = countvectorsmodel.vocabulary.toList
      vocalList.foreach(v => {
        val index = tempRow.indexOf(vocalList.indexOf(v))
        if(!index.equals(-1)) {
          map = map + (v -> index)
          //println("word: " + v + " indices: " + index)
        }
      })
    })
    map.foreach(f => println("map value "+f._1+":"+f._2))
/*
    idf.select("features").rdd.take(3).foreach(f =>
      {
        f.getAs[SparseVector](0).indices.foreach(x => {
          //println("value "+x)
        })
    }
    )
    //println(idf.select("features").rdd.collect().toList.foreach[SparseVector](f => f))
    //val i = 0
    //println(idf.select("features").rdd.map(x => (x, incrementFunction(i))))
    //idf.select("features").rdd.map(

    //idf.select("features").rdd.collect().foreach[SparseVector](x => ))

    /**
      * Work on CountVectorizer
      *
      * Experiment with changing the value of the vocabSize
      * Experiment with changing the value of minDF
      */


    /**
      * alternatively, define CountVectorizerModel with a-priori vocabulary
      */
*/

    spark.stop()
  }
  private def incrementFunction (i : Int) : Int = {
    return i + 1
  }
}
// scalastyle:on println
