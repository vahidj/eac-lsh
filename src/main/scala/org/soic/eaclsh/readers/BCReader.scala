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
import org.apache.spark.sql.SparkSession

/**
  * Created by vjalali on 3/19/16.
  */

class BCReader extends Reader{
    def Output(indexed: DataFrame): DataFrame= {
    val transformedDf = indexed.drop("clump_thickness").
      drop("u_cell_size").
      drop("u_cell_shape").
      drop("marginal_adhesion").
      drop("single_epithelial").
      drop("bare_nuclei").
      drop("bland_chromatin").
      drop("normal_nucleoli").
      drop("mitoses").
      drop("class")
      var assembler = new VectorAssembler().setInputCols(Array("clump_thicknessIndex", "u_cell_sizeIndex", "u_cell_shapeIndex",
        "marginal_adhesionIndex", "single_epithelialIndex", "bare_nucleiIndex",
        "bland_chromatinIndex", "normal_nucleoliIndex", "mitosesIndex"))
      .setOutputCol("features")
      var output = assembler.transform(transformedDf)
      return output
  }

  def DFTransformed(indexed: DataFrame): RDD[LabeledPoint] = {
    val spark = SparkSession
    .builder()
    .getOrCreate()     
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._     
    val transformed = indexed.map(x => new LabeledPoint(x.get(19).asInstanceOf[Double],
      new DenseVector(Array(x.get(10).asInstanceOf[Double], x.get(11).asInstanceOf[Double], x.get(12).asInstanceOf[Double],
        x.get(13).asInstanceOf[Double], x.get(14).asInstanceOf[Double], x.get(15).asInstanceOf[Double],
        x.get(16).asInstanceOf[Double], x.get(17).asInstanceOf[Double], x.get(18).asInstanceOf[Double]))))
    return transformed.toJavaRDD
  }
  def Indexed(FilePath:String, sc: SparkContext): DataFrame= {
    val rawData = sc.textFile(FilePath)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema = StructType(dataSchema.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9)))
    val carDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("clump_thickness").setOutputCol("clump_thicknessIndex").fit(carDataFrame)
    var indexed = indexer.transform(carDataFrame)
    indexer = new StringIndexer().setInputCol("u_cell_size").setOutputCol("u_cell_sizeIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("u_cell_shape").setOutputCol("u_cell_shapeIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("marginal_adhesion").setOutputCol("marginal_adhesionIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("single_epithelial").setOutputCol("single_epithelialIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("bare_nuclei").setOutputCol("bare_nucleiIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("bland_chromatin").setOutputCol("bland_chromatinIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("normal_nucleoli").setOutputCol("normal_nucleoliIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("mitoses").setOutputCol("mitosesIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("class").setOutputCol("label").fit(indexed)
    indexed = indexer.transform(indexed)
    return indexed
  }

  override def numberOfClasses: Int = 2

  override def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((0,10),(1,10),(2,10),(3,10),
    (4,10),(5,10),(6,10),(7,10),(8,9))

  override def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]()
  
  override def numericalFeaturesRange: Map[Int, (Double, Double)] = Map[Int, (Double, Double)]()

  override def dataSchema: String = "clump_thickness u_cell_size u_cell_shape marginal_adhesion single_epithelial bare_nuclei bland_chromatin normal_nucleoli mitoses class"

  override def inputFileName: String = "breastcancer/bcCleaned.data"
}
