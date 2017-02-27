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

class ActivityReader extends Reader {
   def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").zipWithIndex.map
    {case (fieldName, i) =>
      StructField(fieldName, if (numericalFeaturesInfo.keySet.contains(i)) DoubleType else StringType, true)})
    val rowRDD = rawData.map(_.split(",")).map(p =>  Row(p(0).toDouble,p(1).toDouble, p(2).toDouble, p(3),p(4),
      p(5), p(6)) )
    val activityDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex").fit(activityDataFrame)
    var indexed = indexer.transform(activityDataFrame)
    indexer = new StringIndexer().setInputCol("model").setOutputCol("modelIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("device").setOutputCol("deviceIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("gt").setOutputCol("gtIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }
   def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("user").
      drop("model").
      drop("device").
      drop("gt")
      var assembler = new VectorAssembler().setInputCols(Array("x","y", "z",
        "userIndex", "modelIndex", "deviceIndex"
        ))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }
   
   def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(10).asInstanceOf[Double],
      new DenseVector(Array(x.get(0).asInstanceOf[Double],x.get(1).asInstanceOf[Double],x.get(2).asInstanceOf[Double],
        x.get(7).asInstanceOf[Double], x.get(8).asInstanceOf[Double], x.get(9).asInstanceOf[Double]
        ))))
    return transformed
  }

  override def numberOfClasses: Int = 7

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((3,9),(4,3),(5,6))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]((0,0.4472657),(1,0.449882),(2,0.5141587))
  
  override def numericalFeaturesRange: Map[Int, (Double, Double)] = Map[Int, (Double, Double)]((0, (0,1)),(1, (0,1)),(2, (0,1)) )

  override def dataSchema: String = "x y z user model device gt"

  override def inputFileName: String = "activity/activityCleaned.data"
}