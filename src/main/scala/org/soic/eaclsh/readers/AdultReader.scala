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

// reads adult data set
class AdultReader extends Reader {
   def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").zipWithIndex.map
    {case (fieldName, i) =>
      StructField(fieldName, if (numericalFeaturesInfo.keySet.contains(i)) DoubleType else StringType, true)})
    val rowRDD = rawData.map(_.split(",")).map(p =>  Row(p(0).toDouble,p(1), p(2).toDouble, p(3),p(4).toDouble,
      p(5), p(6), p(7), p(8), p(9),
      p(10).toDouble, p(11).toDouble, p(12).toDouble, p(13), p(14)))
    val adultDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("workclass").setOutputCol("workclassIndex").fit(adultDataFrame)
    var indexed = indexer.transform(adultDataFrame)
    indexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupationIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("relationship").setOutputCol("relationshipIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("race").setOutputCol("raceIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("country").setOutputCol("countryIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("income").setOutputCol("label").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }
   def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("workclass").
      drop("education").
      drop("marital").
      drop("occupation").
      drop("relationship").
      drop("race").
      drop("sex").
      drop("country").
      drop("income")
      var assembler = new VectorAssembler().setInputCols(Array("age","workclassIndex", "fnlwgt",
        "educationIndex", "education-num", "maritalIndex", "occupationIndex",
        "relationshipIndex", "raceIndex", "sexIndex", "capital-gain", "capital-loss",
        "hours-per-week","countryIndex"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }
   
   def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val transformed = indexed.map(x => new LabeledPoint(x.get(23).asInstanceOf[Double],
      new DenseVector(Array(x.get(0).asInstanceOf[Double],x.get(15).asInstanceOf[Double],x.get(2).asInstanceOf[Double],
        x.get(16).asInstanceOf[Double], x.get(4).asInstanceOf[Double], x.get(17).asInstanceOf[Double],
        x.get(18).asInstanceOf[Double], x.get(19).asInstanceOf[Double], x.get(20).asInstanceOf[Double],
        x.get(21).asInstanceOf[Double], x.get(10).asInstanceOf[Double], x.get(11).asInstanceOf[Double],
        x.get(12).asInstanceOf[Double], x.get(22).asInstanceOf[Double]))))
    return transformed
  }

  override def numberOfClasses: Int = 2

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((1,7),(3,16),(5,7),(6,14),(7,6),(8,5),(9,2),(14,41))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]((0,13.13466),(2,105653),(4,2.549995),(10,7406.346),(11,404.2984),(12,11.97998))

  override def dataSchema: String = "age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week country income"

  override def inputFileName: String = "adult/adultCleaned.data"
}