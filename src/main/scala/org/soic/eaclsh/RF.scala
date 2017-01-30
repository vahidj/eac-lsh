package org.soic.eaclsh

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
import org.soic.eaclsh.readers._


/**
  * Created by vjalali on 2/27/16.
  */
object RF {
  System.setProperty("hadoop.home.dir", "c:/winutil")
  def makeLibSVMLine(line: String): String =
  {
    val fields = line.split(",")
    return fields(6).toString + " 1:" + fields(0).toString + " 2:" + fields(1).toString +
      " 3:" + fields(2).toString + " 4:" + fields(3).toString + " 5:" + fields(4).toString + " 6:" + fields(5).toString
  }

  def main(args: Array[String]) = {
    val sc: SparkContext = new SparkContext()
    val filePathAdult="/Users/vjalali/Documents/Acad/eac/datasets/adult/adult.data"
    val filePathCar= "/Users/vjalali/Documents/Acad/eac/datasets/careval/car.data"
    val schemaStringAdult = "age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week country income"
    val schemaStringCar= "buying maint doors persons lug_boot safety acceptability"
    val readr= new CarReader // new adultReader
    val indexed = readr.Indexed(filePathCar,sc)
    val transformed = readr.DFTransformed(indexed)
    val output = readr.Output(indexed)
    
    /*val rawData = sc.textFile("/Users/vjalali/Documents/Acad/eac/datasets/careval/car.data")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schemaString = "buying maint doors persons lug_boot safety acceptability"
    val schema = StructType(schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
    val rowRDD = rawData.map(_.split(",")).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6)))
    val carDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    var indexer = new StringIndexer().setInputCol("buying").setOutputCol("buyingIndex").fit(carDataFrame)
    var indexed = indexer.transform(carDataFrame)
    indexer = new StringIndexer().setInputCol("maint").setOutputCol("maintIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("doors").setOutputCol("doorsIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("persons").setOutputCol("personsIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("lug_boot").setOutputCol("lug_bootIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("safety").setOutputCol("safetyIndex").fit(indexed)
    indexed = indexer.transform(indexed)
    indexer = new StringIndexer().setInputCol("acceptability").setOutputCol("label").fit(indexed)
    indexed = indexer.transform(indexed)

    val transformedDf = indexed.drop("buying").
      drop("maint").
      drop("doors").
      drop("persons").
      drop("lug_boot").
      drop("safety").
      drop("acceptability")

    var assembler = new VectorAssembler().setInputCols(Array("buyingIndex", "maintIndex", "doorsIndex", "personsIndex", "lug_bootIndex", "safetyIndex"))
      .setOutputCol("features")

    var output = assembler.transform(transformedDf)
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    output.foreach(x => println(x.toString()))
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val transformed = indexed.map(x => new LabeledPoint(x.get(13).asInstanceOf[Double],
      new DenseVector(Array(x.get(7).asInstanceOf[Double], x.get(8).asInstanceOf[Double], x.get(9).asInstanceOf[Double],
        x.get(10).asInstanceOf[Double], x.get(11).asInstanceOf[Double], x.get(12).asInstanceOf[Double]))))
    */
    //transformed.foreach(x => println(x.label))

    val splits = transformed.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    //trainingData.foreach(x => println(x.get(0).asInstanceOf[String]))
    //trainingData.map(x => new LabeledPoint(x.get(7).asInstanceOf[Double], new DenseVector(Array(0.2))))
    //val traidningRdd = trainingData.javaRDD.map(row => new LabeledPoint(row.toString.spli
    val numClasses = 4
    val categoricalFeaturesInfo = Map[Int, Int]((0,4),(1,4),(2,4),(3,3),(4,3),(5,3))
    val numTrees = 100 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val nfolds: Int = 10
    val rf = new RandomForestClassifier()

    /*val rf = new RandomForestClassifier().setFeaturesCol("buyingIndex")
      .setFeaturesCol("maintIndex")
      .setFeaturesCol("doorsIndex")
      .setFeaturesCol("personsIndex")
      .setFeaturesCol("lug_bootIndex")
      .setFeaturesCol("safetyIndex")
      .setLabelCol("acceptabilityIndex")*/

    val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(1,5,10)).addGrid(rf.maxDepth, Array(2,4,6))
      .addGrid(rf.maxBins, Array(30,60)).build()
    //val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(1,5,10,30,60,90)).addGrid(rf.maxDepth, Array(1,2,3,4,5,6,7,8,9,10))
    //  .addGrid(rf.maxBins, Array(30, 60, 90)).build()
    /*val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)*/

    val cv = new CrossValidator().setEstimator(rf).setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(nfolds)

    //val model = RandomForest.trainClassifier(trainingData.toJavaRDD(),
    //  numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 10)

    //println("++++++++++++++++++++++++++++++++++++++++\n"+cv.fit(output).bestModel.params.toString())
    cv.fit(output).bestModel.params.foreach(x => println(x))
    
    
    // Extracting best model params
    
    val cvModel= cv.fit(output)
    val paramMap = {cvModel.getEstimatorParamMaps
           .zip(cvModel.avgMetrics)
           .maxBy(_._2)
           ._1}
    println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    println("Best Model params\n"+ paramMap.toSeq.filter(_.param.name == "maxBins")(0).value)

    //paramMap.toSeq.filter(_.param.name == "maxBins")(0).value
    
    val model = RandomForest.trainClassifier(trainingData.toJavaRDD(),
      numClasses, categoricalFeaturesInfo,
      paramMap.toSeq.filter(_.param.name == "numTrees")(0).value.asInstanceOf[Int],
      featureSubsetStrategy, impurity,
      paramMap.toSeq.filter(_.param.name == "maxDepth")(0).value.asInstanceOf[Int],
      paramMap.toSeq.filter(_.param.name == "maxBins")(0).value.asInstanceOf[Int], 10)


    val labelAndPreds = testData.map{
      point => val prediction = model.predict(point.features)
        //println(point.label + " " + prediction)
        (point.label, prediction)
    }

    //println(labelAndPreds.filter(r => r._1 != r._2).count())
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count() * 1.0/testData.count()

    println("Test Error = " + testErr)
    //println("Learned classification forest model:\n" + model.toDebugString)
    /*val data = rawData.map{line =>
        val values = line.split(",")
        val featureVector = Vectors.dense(1)
        val label = 2
        LabeledPoint(label, featureVector)
    }*/
    //MLUtils.
    //val data = MLUtils.loadLibSVMFile(sc, "/Users/vjalali/Documents/Acad/eac/datasets/careval/car.data")
    //data.foreach(println(_))
  }
}
