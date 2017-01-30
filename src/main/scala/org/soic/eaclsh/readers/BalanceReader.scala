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
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
/**
  * Created by vjalali on 3/17/16.
  */
class BalanceReader extends Reader{
  def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("class")
      .drop("left-weight").drop("left-distance").drop("right-weight").drop("right-distance")
      var assembler = new VectorAssembler().setInputCols(Array("left-weightIndex", "left-distanceIndex", "right-weightIndex", "right-distanceIndex"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }

  def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(5).asInstanceOf[Double],
      new DenseVector(Array(x.get(6).asInstanceOf[Double], x.get(7).asInstanceOf[Double], x.get(8).asInstanceOf[Double],
        x.get(9).asInstanceOf[Double]))))
    return transformed
  }
  def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4)))
    val balanceDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("class").setOutputCol("classIndex").fit(balanceDataFrame)
    var indexed = indexer.transform(balanceDataFrame)
    indexer = new StringIndexer().setInputCol("left-weight").setOutputCol("left-weightIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("left-distance").setOutputCol("left-distanceIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("right-weight").setOutputCol("right-weightIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("right-distance").setOutputCol("right-distanceIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }

  override def numberOfClasses: Int = 3

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((0,5),(1,5),(2,5),(3,5))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]()

  override def dataSchema: String = "class left-weight left-distance right-weight right-distance"

  override def inputFileName: String = "balance/balance-scale.data"
}
