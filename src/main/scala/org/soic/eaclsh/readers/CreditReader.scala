package org.soic.eaclsh.readers

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
/**
  * Created by vjalali on 3/20/16.
  */
class CreditReader extends Reader{
   def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").zipWithIndex.map
    {case (fieldName, i) =>
      StructField(fieldName, if (numericalFeaturesInfo.keySet.contains(i)) DoubleType else StringType, true)})
    val rowRDD = rawData.map(_.split(",")).map(p =>  Row(p(0),p(1).toDouble, p(2).toDouble, p(3), p(4),
      p(5), p(6), p(7).toDouble, p(8), p(9), p(10).toDouble, p(11), p(12), p(13).toDouble, p(14).toDouble, p(15)))
    val adultDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("a1").setOutputCol("a1Index").fit(adultDataFrame)
    var indexed = indexer.transform(adultDataFrame)
    indexer = new StringIndexer().setInputCol("a4").setOutputCol("a4Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a5").setOutputCol("a5Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a6").setOutputCol("a6Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a7").setOutputCol("a7Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a9").setOutputCol("a9Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a10").setOutputCol("a10Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a12").setOutputCol("a12Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a13").setOutputCol("a13Index").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("a16").setOutputCol("label").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }

  def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("a1").
      drop("a4").
      drop("a5").
      drop("a6").
      drop("a7").
      drop("a9").
      drop("a10").
      drop("a12").
      drop("a13").
      drop("a16")
      var assembler = new VectorAssembler().setInputCols(Array("a1Index","a2", "a3",
        "a4Index", "a5Index", "a6Index", "a7Index",
        "a8", "a9Index", "a10Index", "a11", "a12Index",
        "a13Index","a14","a15"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }

   def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(25).asInstanceOf[Double],
      new DenseVector(Array(x.get(16).asInstanceOf[Double],x.get(1).asInstanceOf[Double],x.get(2).asInstanceOf[Double],
        x.get(17).asInstanceOf[Double], x.get(18).asInstanceOf[Double], x.get(19).asInstanceOf[Double],
        x.get(20).asInstanceOf[Double], x.get(7).asInstanceOf[Double], x.get(21).asInstanceOf[Double],
        x.get(22).asInstanceOf[Double], x.get(10).asInstanceOf[Double], x.get(23).asInstanceOf[Double],
        x.get(24).asInstanceOf[Double], x.get(13).asInstanceOf[Double], x.get(14).asInstanceOf[Double]))))
    return transformed
  }

  override def numberOfClasses: Int = 2

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((0,2),(3,3),(4,3),(5,14),(6,9),(8,2),(9,2),(11,2),(12,3))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]((1,11.83827),(2,5.027077),(7,3.37112),(10,4.968497),(13,168.2968),(14,5253.279))
  
  override def numericalFeaturesRange: Map[Int, (Double, Double)] = Map[Int, (Double, Double)]()

  override def dataSchema: String = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16"

  override def inputFileName: String = "credit/crxCleaned.data"
}
