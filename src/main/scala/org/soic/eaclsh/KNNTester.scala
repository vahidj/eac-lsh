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
    val sc: SparkContext = new SparkContext()
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
    val readr = new CarReader // new adultReader
    //val readr = new CreditReader
    val indexed = readr.Indexed(EACLshConfig.BASEPATH + "dataset/" + readr.inputFileName /*filePathBalance*//*filePathCar*/ /*schemaStringBalance*/ /*schemaStringCar*/,sc)
    var transformed = readr.DFTransformed(indexed)
    //val output = readr.Output(indexed)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val pw = new PrintWriter(new File("results_car.txt"))
    for (i <- 0 until 1) {
      val splits = transformed.randomSplit(Array(0.7, 0.3))
      val (trainingData, testData) = (splits(0), splits(1))
      /*val schema = StructType([
        StructField("label", DoubleType, true),
      StructField("features", VectorUDT, true)
      ])*/
      val schema = StructType("features initial_label".split(" ").zipWithIndex.map
        {case (fieldName, i) =>
      if (i==0) StructField(fieldName, new VectorUDT(), true) else StructField(fieldName, DoubleType, true) })
      val output2 = sqlContext.createDataFrame(trainingData.map(r => {
        Row(r.features, r.label)
      }), schema)

      var indexer = new StringIndexer().setInputCol("initial_label").setOutputCol("label").fit(output2)
      var output = indexer.transform(output2)

//      val rf = new RandomForestClassifier()
//      val paramGrid_rf= new ParamGridBuilder().addGrid(rf.numTrees, Array(2,5,10,20,100)).addGrid(rf.maxDepth, Array(2,4,6,8))
//      .addGrid(rf.maxBins, Array(120)).addGrid(rf.impurity, Array("entropy", "gini")).build()
//      val nfolds: Int = 10
//      val cv_rf  = new CrossValidator().setEstimator(rf).setEvaluator(new MulticlassClassificationEvaluator())
//      .setEstimatorParamMaps(paramGrid_rf)
//      .setNumFolds(nfolds)
//      
//      val cvModel_rf= cv_rf.fit(output)
//      val MaxDepth= cvModel_rf.getEstimatorParamMaps
//           .zip(cvModel_rf.avgMetrics)
//           .maxBy(_._2)
//           ._1.getOrElse(rf.maxDepth, 0)
//      
//      val NumTrees= cvModel_rf.getEstimatorParamMaps
//           .zip(cvModel_rf.avgMetrics)
//           .maxBy(_._2)
//           ._1.getOrElse(rf.numTrees, 0)
//      
//      val MaxBins= cvModel_rf.getEstimatorParamMaps
//           .zip(cvModel_rf.avgMetrics)
//           .maxBy(_._2)
//           ._1.getOrElse(rf.maxBins, 0)
//      
//      val Impurity= cvModel_rf.getEstimatorParamMaps
//           .zip(cvModel_rf.avgMetrics)
//           .maxBy(_._2)
//           ._1.getOrElse(rf.impurity, "null")
//      val featureSubsetStrategy = "auto"  
//      val model_rf = RandomForest.trainClassifier(trainingData.toJavaRDD(),
//        readr.numberOfClasses, readr.categoricalFeaturesInfo, NumTrees, featureSubsetStrategy, Impurity, MaxDepth, MaxBins, 10)
//   
//           
//      println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
//      println("Random Forest Best Params\n"+ "max dept" +MaxDepth+ "max bins\n"+ MaxBins+ "impurity\n"+ Impurity)
      

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

      //val numClasses = 4
      //val categoricalFeaturesInfo = Map[Int, Int]((0,4),(1,4),(2,4),(3,3),(4,3),(5,3))
      //val numTrees = 100 // Use more in practice.
      //val featureSubsetStrategy = "auto" // Let the algorithm choose.
     // val impurity = "gini"
     // val maxDepth = 5
     // val maxBins = 32

      //trainingData.saveAsTextFile("train")
      //testData.saveAsTextFile("test")
      //val tmp: RDD[String] = sc.textFile(EACLshConfig.BASEPATH + "train.txt")
      //println(tmp.count().asInstanceOf[Int])
      //tmp.foreach(r => println(r.toString))
      //println("+++++++++++++++++++++++++++++++++++" + tmp.toString())

     // val nfolds: Int = 20
	  val best_params = List(10, 10, 10)
      val knn = new EACLsh(best_params(0), best_params(1), best_params(2),
       trainingData, testData, readr.categoricalFeaturesInfo, readr.numericalFeaturesInfo, true)
      //val neighbors = testData.zipWithIndex().map{case (k, v) => (v, k)}
      //  .map(r => (r._1.asInstanceOf[Int], knn.getSortedNeighbors(r._2.features)))

      //neighbors.saveAsTextFile("neighbors")
      //val paramGrid = new ParamGridBuilder().addGrid(knn.k, Array(1,2,3,4,5,6,7)).build()
      //val paramGrid = new ParamGridBuilder().addGrid(rf.numTrees, Array(1,5,10,30,60,90)).addGrid(rf.maxDepth, Array(1,2,3,4,5,6,7,8,9,10))
      //  .addGrid(rf.maxBins, Array(30, 60, 90)).build()
      /*val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)*/

      //val cv = new CrossValidator().setEstimator(knn).setEvaluator(new MulticlassClassificationEvaluator())
      //  .setEstimatorParamMaps(paramGrid)
      //  .setNumFolds(nfolds)

     // val model = RandomForest.trainClassifier(trainingData.toJavaRDD(),
      //  readr.numberOfClasses, readr.categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 10)

//      var boostingStrategy = BoostingStrategy.defaultParams("Classification")
//      boostingStrategy.setNumIterations(3)
//      boostingStrategy.treeStrategy.setNumClasses(2)
//      boostingStrategy.treeStrategy.setMaxDepth(5)
      //boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

      //val gbModel = GradientBoostedTrees.train(trainingData, boostingStrategy)
//      val lrModel = new LogisticRegressionWithLBFGS().setNumClasses(readr.numberOfClasses).run(trainingData)
//      val nbModel = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")
//      val numIterations = 100
      //val svmModel = SVMWithSGD.train(trainingData, numIterations)
      //svmModel.clearThreshold()
      //println("++++++++++++++++++++++++++++++++++++++++\n"+cv.fit(output).bestModel.params.toString())
      //cv.fit(output).bestModel.params.foreach(x => println(x))


      // Extracting best model params

      //val cvModel= cv.fit(output)
      //val paramMap = {cvModel.getEstimatorParamMaps
      //       .zip(cvModel.avgMetrics)
      //       .maxBy(_._2)
      //       ._1}
      //println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      //println("Best Model params\n"+ paramMap)

      //paramMap.toSeq.filter(_.param.name == "maxBins")(0).value

      knn.train()

      /*println(knn.predict(testData.first().features))
    val labelAndPreds = testData.map{
      point => val prediction = knn.predict(point.features)
        //println(point.label + " " + prediction)
        (point.label, prediction)
    }*/
      //val labelAndPreds = knn.getPredAndLabels()
      //println(testData.count().asInstanceOf[Int])
//      val labeleAndPredsRF = testData.map {
//        point => val prediction = model_rf.predict(point.features)
//          (point.label, prediction)
//      }

      /*val labeleAndPredsGB = testData.map {
        point => val prediction = gbModel.predict(point.features)
          (point.label, prediction)
      }*/

//      val labeleAndPredsLR = testData.map {
//        point => val prediction = lrModel.predict(point.features)
//          (point.label, prediction)
//      }

//      val labeleAndPredsNB = testData.map {
//        point => val prediction = nbModel.predict(point.features)
//          (point.label, prediction)
//      }

      /*val labeleAndPredsSVM = testData.map{
      point => val prediction = svmModel.predict(point.features)
        (point.label, prediction)
    }*/

      //val labelAndPreds = knn.getPredAndLabels()
//      val predsAndLabelsKnn = knn.getPredAndLabelsKNN()
      
      val predsAndLabelsKnnLsh = knn.getPredAndLabelsKNNLsh()
//      val metrics = new MulticlassMetrics(predsAndLabelsKnnLsh)
//      println("Confusion matrix:")
//      println(metrics.confusionMatrix)
//
////      // Overall Statistics
////      val accuracy = metrics.
////      println("Summary Statistics")
////      println(s"Accuracy = $accuracy")
//      
//      // Precision by label
//      val labels = metrics.labels
//      labels.foreach { l =>
//        println(s"Precision($l) = " + metrics.precision(l))
//      }
//      
//      // Recall by label
//      labels.foreach { l =>
//        println(s"Recall($l) = " + metrics.recall(l))
//      }
//      
//      // False positive rate by label
//      labels.foreach { l =>
//        println(s"FPR($l) = " + metrics.falsePositiveRate(l))
//      }
//      
//      // F-measure by label
//      labels.foreach { l =>
//        println(s"F1-Score($l) = " + metrics.fMeasure(l))
//      }
      
      
      //println(labelAndPreds)
      //println(labelAndPreds.filter(r => r._1 != r._2).count())
//      val testErrKNN = predsAndLabelsKnn.filter(r => r._1 != r._2).length * 1.0 / testData.count()
//      println("here it is: " + testErrKNN)
      
//      val testErr = labelAndPreds.filter(r => r._1 != r._2).length * 1.0 / testData.count()
//      val testErrRF = labeleAndPredsRF.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
//      //val testErrGB = labeleAndPredsGB.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
//      val testErrLR = labeleAndPredsLR.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
//      val testErrNB = labeleAndPredsNB.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0 / testData.count()
//      //val testErrSVM = labeleAndPredsSVM.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0/testData.count()
//      println("EAC Test Error = " + testErr + " RF test error = " + testErrRF + " KNN test error = " + testErrKNN +
//        "  Logistic Regression test error " + testErrLR
//        + " Naive Bayes test error " + testErrNB /*+ " GB test error " + testErrGB + " SVM test error " + testErrSVM*/)
//      pw.write(NumTrees + " " + featureSubsetStrategy + " " + Impurity + " " + MaxDepth + " " + MaxBins + "\n")
//      pw.write(best_params.toString() + "\n")
//      pw.write(testErr + " " + testErrRF + " " + testErrKNN + " " + testErrLR + " " + testErrNB /*+ " " + testErrGB*/)
    }
//    pw.close
    //val testErrRF = labeleAndPredsRF.filter(r => r._1 != r._2).count().asInstanceOf[Int] * 1.0/testData.count()
    //println(testErrRF)
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
