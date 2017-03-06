package org.soic.eaclsh

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.classification.{SVMWithSGD, NaiveBayes, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.soic.eaclsh.EACLshConfig._
import org.soic.eaclsh.readers.CarReader
import java.io._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.VectorUDT
import org.soic.eaclsh.readers.ActivityReader
import org.soic.eaclsh.readers.ActivityReader
import org.soic.eaclsh.readers.BalanceReader
import org.soic.eaclsh.readers.CreditReader
import org.soic.eaclsh.readers.ActivityReader
import org.soic.eaclsh.readers.CarReader
import org.soic.eaclsh.readers.ActivityReader
import org.apache.spark.sql.SparkSession
import org.soic.eaclsh.readers.ActivityReaderNew

/**
  * Created by vjalali on 2/27/16.
  */
object KNNTester {
  System.setProperty("hadoop.home.dir", "c:/winutil")
  def makeLibSVMLine(line: String): String =
  {
    val fields = line.split(",")
    return fields(6).toString + " 1:" + fields(0).toString + " 2:" + fields(1).toString +
      " 3:" + fields(2).toString + " 4:" + fields(3).toString + " 5:" + fields(4).toString + " 6:" + fields(5).toString
  }

  def main(args: Array[String]) = {
    val method = args(0)
    val fileName = args(1)
    val neighbors = args(2).toInt
    
    val arcf = Array(("spark.streaming.unpersist","true"),("spark.executor.memory","24g"),("spark.mesos.coarse","true"), 
        ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"), ("spark.kryoserializer.buffer", "128k"), 
        ("spark.kryoserializer.buffer.max", "2047"), ("spark.rdd.compress", "true"),
        ("spark.cores.max", "128"), ("spark.akka.heartbeat.interval", "10000s"), ("spark.akka.heartbeat.pauses", "15000s"),
        ("spark.driver.memory", "12g"), ("spark.memory.fraction", "0.9"), ("spark.memory.storageFraction", "0.1"),
        ("spark.ui.retainedJobs", "200"), ("spark.ui.retainedStages", "200"), ("spark.yarn.executor.memoryOverhead","2g"), ("spark.driver.maxResultSize","2g"))
    val confak = new SparkConf().setAppName("test").setAll(arcf)
//    val sc = new SparkContext(cf)
    //val confak = new SparkConf().setAppName("test").set("spark.driver.allowMultipleContexts", "true")
    val spark = SparkSession.builder.config(conf=confak).enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext
    //val sc: SparkContext = new SparkContext()
    //val filePathAdult="/Users/vjalali/Documents/Acad/eac/datasets/adult/adult.data"
    println(EACLshConfig.BASEPATH)
    val filePathCar= EACLshConfig.BASEPATH + "datasets/careval/car.data"
    val filePathBalance = EACLshConfig.BASEPATH + "datasets/balance/balance-scale.data"
    val filePathAdult = EACLshConfig.BASEPATH + "datasets/adult/adultCleaned.data"
    val filePathBC = EACLshConfig.BASEPATH + "datasets/breastcancer/bcCleaned.data"
    val filePathBankruptcy = EACLshConfig.BASEPATH + "datasets/bankruptcy/bankruptcy.data"
    val filePathCredit = EACLshConfig.BASEPATH + "datasets/credit/crxCleaned.data"
    val schemaStringAdult = "age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week country income"
    val schemaStringCar= "buying maint doors persons lug_boot safety acceptability"
    val schemaStringBalance = "class left-weight left-distance right-weight right-distance"
    val schemaStringBC = "clump_thickness u_cell_size u_cell_shape marginal_adhesion single_epithelial bare_nuclei bland_chromatin normal_nucleoli mitoses class"
    val schemaStringBankruptcy = "industrial_risk management_risk financial_flexibility credibility competitiveness operating_risk class"
    val schemaStringCredit = "a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16"
//    val readr = new ActivityReader // new adultReader
//    //val readr = new CreditReader
//    val indexed = readr.Indexed(EACLshConfig.BASEPATH + "dataset/" + readr.inputFileName /*filePathBalance*//*filePathCar*/ /*schemaStringBalance*/ /*schemaStringCar*/,sc)
//    var transformed = readr.DFTransformed(indexed)
    
    val readr = ActivityReaderNew
    var transformed = readr.readData(sc, EACLshConfig.BASEPATH + "dataset/activity/" + fileName , "x y z user model device")
    //transformed.take(100).foreach { x => println(x.label + " " + x.features.toString()) }
    //System.exit(1)
    //val output = readr.Output(indexed)
    //val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    
    val cts = System.currentTimeMillis.toString()
    
    val pw = new PrintWriter(new File("results"+cts+".txt"))
    pw.write(readr.getClass.toString() + " " + method + " " + System.currentTimeMillis() + " " + fileName + " #neighbors= " + neighbors + "\n")
    
    for (i <- 0 until 10) {
      //val splits = transformed.randomSplit(Array(0.1, 0.9))(0).randomSplit(Array(0.8, 0.2))
      val splits = transformed.randomSplit(Array(0.8, 0.2))
      val (trainingData, testData) = (splits(0), splits(1))
      //val trainingData = trainingData2.randomSplit(Array(0.01, 0.99))(0)
      //val testData = testData2.randomSplit(Array(0.01, 0.99))(0)
      //trainingData.foreach { x => println(x.toString()) }
      //System.exit(1)
      /*val schema = StructType([
        StructField("label", DoubleType, true),
      StructField("features", VectorUDT, true)
      ])*/
//      val schema = StructType("features initial_label".split(" ").zipWithIndex.map
//        {case (fieldName, i) =>
//      if (i==0) StructField(fieldName, new VectorUDT(), true) else StructField(fieldName, DoubleType, true) })
//      val output2 = sqlContext.createDataFrame(trainingData.map(r => {
//        Row(r.features, r.label)
//      }), schema)
//
//      var indexer = new StringIndexer().setInputCol("initial_label").setOutputCol("label").fit(output2)
//      var output = indexer.transform(output2)

      
      if (method.equals("rf")){
        val rf = new RandomForestClassifier()
//        val paramGrid_rf= new ParamGridBuilder().addGrid(rf.numTrees, Array(2,5,10,20,100)).addGrid(rf.maxDepth, Array(2,4,6,8))
//        .addGrid(rf.maxBins, Array(120)).addGrid(rf.impurity, Array("entropy", "gini")).build()
//        val nfolds: Int = 10
//        val cv_rf  = new CrossValidator().setEstimator(rf).setEvaluator(new MulticlassClassificationEvaluator())
//        .setEstimatorParamMaps(paramGrid_rf)
//        .setNumFolds(nfolds)
//        
//        val cvModel_rf= cv_rf.fit(trainingData.)
//        val MaxDepth= cvModel_rf.getEstimatorParamMaps
//             .zip(cvModel_rf.avgMetrics)
//             .maxBy(_._2)
//             ._1.getOrElse(rf.maxDepth, 0)
//        
//        val NumTrees= cvModel_rf.getEstimatorParamMaps
//             .zip(cvModel_rf.avgMetrics)
//             .maxBy(_._2)
//             ._1.getOrElse(rf.numTrees, 0)
//        
//        val MaxBins= cvModel_rf.getEstimatorParamMaps
//             .zip(cvModel_rf.avgMetrics)
//             .maxBy(_._2)
//             ._1.getOrElse(rf.maxBins, 0)
//        
//        val Impurity= cvModel_rf.getEstimatorParamMaps
//             .zip(cvModel_rf.avgMetrics)
//             .maxBy(_._2)
//             ._1.getOrElse(rf.impurity, "null")
//        val featureSubsetStrategy = "auto"  
//        val model_rf = RandomForest.trainClassifier(trainingData.toJavaRDD(),
//          readr.numberOfClasses, readr.categoricalFeaturesInfo, NumTrees, featureSubsetStrategy, Impurity, MaxDepth, MaxBins, 10)
        
        val model_rf = RandomForest.trainClassifier(trainingData.toJavaRDD(),
          readr.numberOfClasses, readr.categoricalFeaturesInfo2, 10, "auto", "entropy", 4, 100, 10)
        
        val predAndLabelRF = testData.map {
          point => val prediction = model_rf.predict(point.features)
            (prediction, point.label)
        }
        var resList:List[Double] = List[Double]()
        val err = predAndLabelRF.filter(f => f._1 != f._2).count()
        //println("still alive 2")
        resList = err :: resList
        val metrics = new MulticlassMetrics(predAndLabelRF)
        // Weighted stats
        resList = metrics.weightedPrecision :: resList 
        resList = metrics.weightedRecall :: resList
        resList = metrics.weightedFMeasure :: resList
        resList = metrics.weightedFalsePositiveRate :: resList
        resList = metrics.weightedTruePositiveRate :: resList
        pw.write(resList.reverse.mkString("\t") + "\n")        
      }
      else if (method.equals("eaclsh"))
      {
	      val best_params = List(10, 10, 10)
        val knn = new EACLsh(best_params(0), best_params(1), best_params(2),
        trainingData, testData, readr.categoricalFeaturesInfo, readr.numericalFeaturesInfo, readr.numericalFeaturesRange)  
	      knn.train(sc)
        val predsAndLabelsLsh = knn.getPredAndLabelsLshRetarded() //knn.getPredAndLabelsKNNLsh()//knn.getPredAndLabelsLshRetarded()
        var resList:List[Double] = List[Double]()
        val err = predsAndLabelsLsh.filter(f => f._1 != f._2).count()
        resList = err :: resList
        val metrics = new MulticlassMetrics(predsAndLabelsLsh)
        resList = metrics.weightedPrecision :: resList 
        resList = metrics.weightedRecall :: resList
        resList = metrics.weightedFMeasure :: resList
        resList = metrics.weightedFalsePositiveRate :: resList
        resList = metrics.weightedTruePositiveRate :: resList
        pw.write(resList.reverse.mkString("\t") + "\n")	      
      }
      else if (method.equals("knnlsh")){
	      val best_params = List(10, 10, 10)
	      println("==================================INITIALIZATION================================")
        val knn = new EACLsh(neighbors, best_params(1), best_params(2),
        trainingData, testData, readr.categoricalFeaturesInfo, readr.numericalFeaturesInfo, readr.numericalFeaturesRange)
	      //pw.write(System.currentTimeMillis() + " started training \n")
	      println("==================================TRAIN================================")
	      knn.train(sc)
	      //pw.write(System.currentTimeMillis() + " started prediction \n")
	      println("==================================PREDICTION================================")
        val predsAndLabelsLsh = knn.getPredAndLabelsKNNLshOld(sc) //knn.getPredAndLabelsKNNLsh()//knn.getPredAndLabelsLshRetarded()
        println("==================================PREDICTION FINISHED================================" + predsAndLabelsLsh.count())
        //pw.write(System.currentTimeMillis() + " ended prediction \n")
        var resList:List[Double] = List[Double]()
        val err = predsAndLabelsLsh.filter(f => f._1 != f._2).count()
        resList = err :: resList
        val metrics = new MulticlassMetrics(predsAndLabelsLsh)
        resList = metrics.weightedPrecision :: resList 
        resList = metrics.weightedRecall :: resList
        resList = metrics.weightedFMeasure :: resList
        resList = metrics.weightedFalsePositiveRate :: resList
        resList = metrics.weightedTruePositiveRate :: resList
        pw.write(resList.reverse.mkString("\t") + "\n")
      }
      else if (method.equals("nb")){
        val nbModel = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")
        val predsAndLabels = testData.map {
          point => val prediction = nbModel.predict(point.features)
          (point.label, prediction)
        }
        
        var resList:List[Double] = List[Double]()
        val err = predsAndLabels.filter(f => f._1 != f._2).count()
        resList = err :: resList
        val metrics = new MulticlassMetrics(predsAndLabels)
        resList = metrics.weightedPrecision :: resList 
        resList = metrics.weightedRecall :: resList
        resList = metrics.weightedFMeasure :: resList
        resList = metrics.weightedFalsePositiveRate :: resList
        resList = metrics.weightedTruePositiveRate :: resList
        pw.write(resList.reverse.mkString("\t") + "\n")
      }
      else if (method.equals("lr")){
        val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(readr.numberOfClasses).run(trainingData)
        val predsAndLabels = testData.map {
          point => val prediction = lrModel.predict(point.features)
          (point.label, prediction)
        }
        
        var resList:List[Double] = List[Double]()
        val err = predsAndLabels.filter(f => f._1 != f._2).count()
        resList = err :: resList
        val metrics = new MulticlassMetrics(predsAndLabels)
        resList = metrics.weightedPrecision :: resList 
        resList = metrics.weightedRecall :: resList
        resList = metrics.weightedFMeasure :: resList
        resList = metrics.weightedFalsePositiveRate :: resList
        resList = metrics.weightedTruePositiveRate :: resList
        pw.write(resList.reverse.mkString("\t") + "\n")
      }
      
      
      

      /*val neighbor_nos = List(1, 2, 3, 5, 10)
      val rule_nos = List(1, 2, 3, 5, 10)
      val rule_learning_nos = List(5, 10, 20)
      var best_params = List(0, 0 , 0)
      var min_err = 100.0
      neighbor_nos.foreach(a1 => {
        rule_nos.foreach(a2 => {
          rule_learning_nos.foreach(a3 => {


            val cv_splits = MLUtils.kFold(trainingData, 10, 10)
            //println(cv_splits.foreach(r => println("r " + r.toString())))
            cv_splits.zipWithIndex.foreach{case ((cv_training, cv_validation), splitIndex) =>
               val cv_eac = new EAC(a1, a2, a3, cv_training, cv_validation, readr.categoricalFeaturesInfo,
                 readr.numericalFeaturesInfo)
               cv_eac.train()
               val cv_labelAndPreds = cv_eac.getPredAndLabels()
               val cv_testErr = cv_labelAndPreds.filter(r => r._1 != r._2).length * 1.0 / cv_validation.count()
               if (cv_testErr < min_err) {
                 min_err = cv_testErr
                 best_params = List(a1, a2, a3)
               }

            }
			//println(best_params.toString)
			//System.exit(0)
          })
        })
      })*/





    }
    pw.write(System.currentTimeMillis() +"\n")
    pw.close()
  }
}