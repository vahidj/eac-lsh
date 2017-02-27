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
  * Created by vjalali on 3/20/16.
  */
class BankruptcyReader extends Reader{
  def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("industrial_risk").
      drop("management_risk").
      drop("financial_flexibility").
      drop("credibility").
      drop("competitiveness").
      drop("operating_risk").
      drop("class")
      var assembler = new VectorAssembler().setInputCols(Array("industrial_riskIndex", "management_riskIndex",
        "financial_flexibilityIndex", "credibilityIndex", "competitivenessIndex", "operating_riskIndex"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }

  def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(13).asInstanceOf[Double],
      new DenseVector(Array(x.get(7).asInstanceOf[Double], x.get(8).asInstanceOf[Double], x.get(9).asInstanceOf[Double],
        x.get(10).asInstanceOf[Double], x.get(11).asInstanceOf[Double], x.get(12).asInstanceOf[Double]))))
    return transformed
  }
  def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5)))
    val carDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("industrial_risk").setOutputCol("industrial_riskIndex").fit(carDataFrame)
    var indexed = indexer.transform(carDataFrame)
    indexer = new StringIndexer().setInputCol("management_risk").setOutputCol("management_riskIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("financial_flexibility").setOutputCol("financial_flexibilityIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("credibility").setOutputCol("credibilityIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("competitiveness").setOutputCol("competitivenessIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("operating_risk").setOutputCol("operating_riskIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("class").setOutputCol("label").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }

  override def numberOfClasses: Int = 2

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((0,3),(1,3),(2,3),(3,3),(4,3),(5,3))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]()
  
  override def numericalFeaturesRange: Map[Int, (Double, Double)] = Map[Int, (Double, Double)]()

  override def dataSchema: String = "industrial_risk management_risk financial_flexibility credibility competitiveness operating_risk class"

  override def inputFileName: String = "bankruptcy/bankruptcy.data"
}
