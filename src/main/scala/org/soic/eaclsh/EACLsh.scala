package org.soic.eaclsh

import java.util

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext, SparkConf}
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{MLWritable, Identifiable}
import org.apache.spark.sql.{SQLContext, DataFrame}
import Array.range
import scala.collection._
import scala.collection.mutable.ListBuffer
import scala.util.Random
import org.soic.eaclsh.readers._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg._
import com.github.karlhigley.spark.neighbors.ANN
import com.github.karlhigley.spark.neighbors.ANNModel
import org.apache.spark.broadcast.Broadcast



class EACLsh(private var k: Int, private val rno: Int, private val ruleRadius: Int, data: RDD[LabeledPoint],
          testData: RDD[LabeledPoint], categoricalFeaturesInfo: Map[Int, Int], numericalFeaturesInfo: Map[Int, Double], inputUseLsh:Boolean)
  extends Serializable with Logging {
  def setK(k: Int): EACLsh = {
    this.k = k
    this
  }

  def getK: Int = k
  private val useLsh = inputUseLsh
  //private var data: RDD[LabeledPoint] = null
  private var dataWithIndex: RDD[(Long, LabeledPoint)] = data.zipWithIndex().map{case (k, v) => (v, k)}
  private var testWithIndex: RDD[(Long, LabeledPoint)] = testData.zipWithIndex().map{case (k, v) => (v, k)}
  private val dataWithIndexList: List[(Long, LabeledPoint)] = dataWithIndex.collect().toList
  private val distThresh: Double = 1.05
  private val ruleDistThresh: Double = 1.3
  private var ruleHyperPlanes:List[List[(Double, Double)]] = null
  //private val uniqs: Array[Array[Double]] = data.map(r => r.features(0)).distinct().collect()

  private var hpNo = 100
  private var annModel: ANNModel = null
  private var annRuleModel: ANNModel = null
  private var testHashedDataset: RDD[(Long, SparseVector)] = null
  //each element in the list contains the distance between pairs of values of the corrsponding feature
  private var mizan = List.fill(this.data.first().features.size)(scala.collection.mutable.Map[(Double, Double),  Double]())
  //private var mizanb : Broadcast[List[Map[(Double, Double), Double]]] = null
  private var ruleMizan = List.fill(this.data.first().features.size)(scala.collection.mutable.Map[((Double, Double), (Double,Double)),  Double]())
  private val ruleBase: RDD[((Double, Double), List[(Double, Double)])] = dataWithIndex.cartesian(dataWithIndex).filter{case (a,b) => a._1 != b._1}
    .map{case ((a,b),(c,d)) => ((b.label, d.label), (b.features.toArray.toList zip d.features.toArray.toList))}
  private val ruleBaseWithIndex = ruleBase.zipWithIndex().map{case (k,v) => (v,k)}
  private var ruleBase4: List[((Double, Double), List[(Double, Double)])] = null
  private var ruleBase4RddIndex: RDD[(Long, ((Double, Double), List[(Double, Double)]))] = null
  private var ruleBase4WithIndex: List[(Int, ((Double, Double), List[(Double, Double)]))] = null
  //private var ruleBase4WithIndex: RDD[(Long, ((Double, Double), List[(Double, Double)]))] = null
  //private var mizan = List[scala.collection.mutable.Map[(Double, Double), Double]]()//List[util.HashMap[(Double, Double), Int]]()
  //2D array, indices represent indices of elements in data, each element represents distances between the case represented by row and column
  private var distances = scala.collection.mutable.Map[(Int, Int), Double]()
  //2D array, indices represent indices of elements in data, each row represents cases in data sorted by their ascending distance to the corresponding case
  private var neighbors = Array.ofDim[Int](data.count().toInt, data.count().toInt - 1)
  private var uniqs: List[Map[Double, Long]] = List[Map[Double, Long]]()
  private var ruleUniqs: List[Map[(Double, Double), Int]] = List[Map[(Double, Double), Int]]()
  
  
  def msort(array: List[Int], baseIndex:Int): List[Int] = {
    val n = array.length/2
    if (n == 0) array
    else{
      def merge(array1: List[Int], array2: List[Int]): List[Int] = (array1, array2) match {
        case (Nil, array2) => array2
        case (array1, Nil) => array1
        case (x :: array11, y :: array21) =>
          if (distances(baseIndex, x) < distances(baseIndex,y)) x :: merge(array11, array2)
          else y :: merge(array1, array21)
      }
      val (left, right) = array splitAt(n)
      merge(msort(left,baseIndex), msort(right,baseIndex))
    }
  }
  def getNearestNeighbors(i:Int): Array[Int] ={
    var result = range(0,data.count().asInstanceOf[Int]).toList
    msort(result, i).toArray[Int]
  }

  def getRuleDistance(r1: List[(Double, Double)], r2: List[(Double, Double)]): Double = {
    var distance: Double = 0
    var featureCounter = 0
    r1.foreach(f1 => {
      val f2 = r2(featureCounter)
      if (categoricalFeaturesInfo.keySet.contains(featureCounter)) {
        if (ruleMizan(featureCounter).contains(f1, f2))
          distance = distance + scala.math.pow(ruleMizan(featureCounter)((f1, f2)), 2)
        else if (ruleMizan(featureCounter).contains(f2, f1))
          distance = distance + scala.math.pow(ruleMizan(featureCounter)((f2, f1)), 2)
        else if (f1 != f2) {
          //println()
          //println(f1 + "   " + f2)
          distance = distance + 1.0 ///(r1.length)
        }
      }
      else{
        distance = distance + math.min(1, Math.abs((f1._1 - f1._2) - (f2._1 - f2._2))/(4 * numericalFeaturesInfo(featureCounter)))
      }
      featureCounter += 1
    })
    math.sqrt(distance)
  }

  /*def getDistance(i:Long, j:Long): Double = {
    var distance: Double = 0
    val c1: LabeledPoint = this.dataWithIndex.lookup(i)(0)
    val c2: LabeledPoint = this.dataWithIndex.lookup(j)(0)
    var featureCounter = 0
    c1.features.toArray.foreach(f1 => {
      val f2 = c2.features.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (mizan(featureCounter).contains(smaller, greater))
        distance = scala.math.pow(mizan(featureCounter)((smaller, greater)), 2)
      featureCounter += 1
    })
    math.sqrt(distance)
  }*/

  def getDistance(c1:Vector, c2:Vector): Double = {
    //println("000000000000000000000000000000zooooor0000000000000000" + mizanb.value.toString())
    //println(c1.toString + "-------------" + c2.toString)
    var distance: Double = 0
	//var test: Int = 0
    var featureCounter = 0
    c1.toArray.foreach(f1 => {
      val f2 = c2.toArray(featureCounter)
      val smaller = Math.min(f1, f2)
      val greater = Math.max(f1,f2)
      if (categoricalFeaturesInfo.keySet.contains(featureCounter)) {
        if (mizan(featureCounter).contains(smaller, greater))
        {
          //println("0000000000000000000gooooor0000000000000000000000000")
          distance = distance + scala.math.pow(mizan(featureCounter)((smaller, greater)), 2)
        }
        else if (smaller != greater) {
          //println(mizan.toString())
          //println(f1 + "    " + f2)
          //System.exit(0)
          distance = distance + 1.0 /// (c1.toArray.length)
        }
      }
      else
        distance = distance + math.min(1, math.abs(f1-f2)/(4 * numericalFeaturesInfo(featureCounter)))
      featureCounter += 1
    })
	//println(test)
	//System.exit(0)
    math.sqrt(distance)
  }
  def getPredAndLabelsLshRetarded(): RDD[(Double,Double)] = {
    val hashedRuleset = ruleBase4RddIndex.map(r => {
      (r._1, getRuleHashBits(r._2._2, ruleHyperPlanes)) } )

    val tmpAnnRuleModel =
      new com.github.karlhigley.spark.neighbors.ANN(dimensions = hpNo, measure = "jaccard")
        .setTables(4)
        .setSignatureLength(128)
        .setPrimeModulus(739)
        .setBands(16)
        .train(hashedRuleset)
        
    val tmp = annModel.neighbors(testHashedDataset, this.k).map(r => {println("00000000000000000000000000000000000000000000000") 
      (r._1, r._2.map(f => f._1))})
    val gharch = tmp.flatMap(f => f._2.map { x => (x, f._1) })
    .join(dataWithIndex).map(f => (f._2._1, f._2._2)).join(testWithIndex)
    .map(f => (f._1,  ((f._2._1.label, f._2._2.label),(f._2._1.features.toArray.toList.zip(f._2._2.features.toArray.toList)))))
    .zipWithIndex().map{case (k, v) => (v, k)}
    
    val gharchNotestInd = gharch
    .map(f => (f._1, f._2._2))
//    .groupByKey()
    val formedRules = gharchNotestInd.map(r => {
      (r._1, getRuleHashBits(r._2._2, ruleHyperPlanes)) } )    

    val tmp2 = tmpAnnRuleModel.neighbors(formedRules, this.rno).map(r =>{println("111111111111111111111111111111111111111111") 
      (r._1, r._2.map(f => f._1))})
    val zaghart = tmp2.flatMap(f => f._2.map { x => (x, f._1) })
    .join(ruleBase4RddIndex).map(f => (f._2._1, f._2._2)).join(gharch)
    .map(f => (f._2._2._1, (f._2._1._1, f._2._2._2._1)))
    .map(f => {
      println("ahahahahahahhahahahahhahahahahahahahah")
      val testInd = f._1
      val pred = if (f._2._1._1 == f._2._2._1) f._2._1._1 else f._2._2._1  
      val lab = f._2._2._2
      (testInd, (pred, lab))
    })
    .groupByKey().map(f => (f._2.toList(0)._2, f._2.toList.map(f => f._1).groupBy(identity).maxBy(_._2.size)._1))
 
    zaghart
  }
  
  def getPredAndLabelsLsh(): List[(Double,Double)] = {
    val tmp = annModel.neighbors(testHashedDataset, this.k).map(r => (r._1, r._2.map(f => f._1)))
    val gharch = tmp.flatMap(f => f._2.map { x => (x, f._1) })
    .join(dataWithIndex).map(f => (f._2._1, f._2._2)).join(testWithIndex).map(f => (f._1, (f._2._1.label, f._2._2.features.toArray.zip(f._2._1.features.toArray))))
    .groupByKey()
    
    var predAndLbls = List[(Double, Double)]()
    for (i <- 0 until testData.count().toInt){
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    1")
      val tmpLabel = gharch.filter(f => f._1 == i).first()._1
      val rulesToConsider = ruleBase4RddIndex.filter(f => f._2._1._1 == tmpLabel)
      val hashedRuleset = rulesToConsider.map(r => {
        (r._1, getRuleHashBits(r._2._2, ruleHyperPlanes)) } )
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    2")
      //println("--------------------------" + hashedRuleset.count())
      //hashedRuleset.foreach(f => println(f.toString()))
        
      val tmpAnnRuleModel =
        new com.github.karlhigley.spark.neighbors.ANN(dimensions = hpNo, measure = "jaccard")
          .setTables(4)
          .setSignatureLength(128)
          .setPrimeModulus(739)
          .setBands(16)
          .train(hashedRuleset)
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    3")
      val hashedRulesToRetrieve = rulesToConsider.map(r => {
        (r._1, getRuleHashBits(r._2._2, ruleHyperPlanes)) } )
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    4") 
      val retrievedRules = tmpAnnRuleModel.neighbors(hashedRulesToRetrieve, rno)
      retrievedRules.cache()
      rulesToConsider.cache()
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    5 retrules " + retrievedRules.count() + " rtocons " + rulesToConsider.count() )
      val zer = retrievedRules.map(r => (r._1, r._2.map(f => f._1))).flatMap(f => f._2.map { x => (x, f._1) })
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    zer")
      val gher = zer
      .join(rulesToConsider).map(f => (f._2._1, f._2._2._1._2)).groupByKey()
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    gher")
      val rel = gher
      .map(f => f._2.toList.groupBy(identity).maxBy(_._2.size)._1).collect().toList.groupBy(identity).maxBy(_._2.size)._1
      println("+++++++++++++++++++++++++++++++++++++++++++++++++++" + i + "    6")
      predAndLbls = predAndLbls ::: List((rel , testWithIndex.filter(f => f._1 == i).first()._2.label))
    }
    predAndLbls
  }
  
  def getPredAndLabelsKNNLsh(): RDD[(Double,Double)] = {
    
    val tmp = annModel.neighbors(testHashedDataset, this.k).map(r => (r._1, r._2.map(f => f._1)))
    //tmp.foreach(f => println(f._1 + " " + f._2.toString()))
    val gharch = tmp.flatMap(f => f._2.map { x => (x, f._1) })
    
    val zerp = gharch.join(dataWithIndex).map(f => (f._2._1, f._2._2.label)).groupByKey().map(f => (f._1, f._2.groupBy(identity).maxBy(_._2.size)._1))
    .join(testWithIndex).map(f => (f._2._1, f._2._2.label))
    
    zerp
//    val zerp = tmp.join(testWithIndex).map(f => (f._1, f._2._1, f._2._2.label))
//    val gharp = zerp.flatMap(f => f._2.map { x => (x, (f._1, f._3)) }).join(dataWithIndex)
//    val tap = gharp.map(f => (f._2._1._1, f._2._1._2 , f._2._2.label))
//    val se = tap.groupBy(f => f._1)
//        
//    
//    
//    tmp.map(f => (testWithIndex.filter(s => s._1 == f._1).first()._2.label ,
//        1.0))    
//    tmp.map(f => (testWithIndex.filter(s => s._1 == f._1).first()._2.label ,
//        f._2.map { x => dataWithIndex.filter(z => z._1 == x).first()._2.label }.groupBy(identity).maxBy(_._2.size)._1))
    
//    annModel.neighbors(testHashedDataset, this.k).map(r => 
//      (r._2.map(f => dataWithIndex.filter(z => z._1 == f._1).first()._2.label).groupBy(identity).maxBy(_._2.size)._1 , 
//          testWithIndex.filter(s => s._1 == r._1).first()._2.label))
  }
      
  def getPredAndLabelsKNN(): List[(Double,Double)] = {
    var result = List[(Double,Double)]()
    testData.collect().foreach(point => result = result ::: List((point.label, predictKNN(point.features))))
    result
    //List((testData.first().label, predict(testData.first().features)))
  }

  
  def getPredAndLabels(): List[(Double,Double)] = {
    var result = List[(Double,Double)]()
    testData.collect().foreach(point => result = result ::: List((point.label, predict(point.features))))
    result
    //List((testData.first().label, predict(testData.first().features)))
  }


  def getTopKWithQSel(inputList: List[(Int, Double)], inputK: Int): List[(Int, Double)] = {
    //print(inputList.toString)
    val pivot = quickSelect(inputList.to[ListBuffer], inputK)
    var result = List[(Int, Double)]()
    inputList.foreach(r => {
      if (r._2 < pivot)
        result = List(r) ::: result
      else if (r._2 == pivot)
        result = result ::: List(r)
    })
    //println(result.toString())
    result.take(inputK)
  }

  def getTopK(inputList: List[(Int, Double)], inputK: Int): List[(Int, Double)] = {
    //println(inputList.toString())
    var result = List[(Int, Double)]()
    inputList.foreach(r => {
      var left = 0
      var right = result.size -1
      if (result.size == 0)
        result = List(r)
      else if (result.size < inputK || result(result.size - 1)._2 > r._2) {
        while (left < right) {
          //println(left + " " + right)
          val mid = left + (right - left) / 2
          if (result(mid)._2 == r._2)
            left = mid
            right = mid
          if (result(mid)._2 < r._2)
            left = mid + 1
          else if (result(mid)._2 > r._2)
            right = mid - 1
        }
        result = result.take(left) ::: List(r) ::: result.drop(left)
        result = result.take(inputK)
      }
    })
    //println("+++++++++++++++++++++++++++++++++++++++++++" + result.size + "++++++++++++++++++++++++++++++++++++++++")
    //println(result.toString())
    result
  }

  def quickSelect(inputList: ListBuffer[(Int, Double)], n: Int, rand: Random = new Random): Double = {
    if (inputList == null)
      throw new Exception();
    var from = 0
    var to = inputList.size - 1
    while(from < to){
      var r = from
      var w = to
      val mid = inputList((r+w)/2)._2
      while(r < w){
        if (inputList(r)._2 >= mid){
          val tmp = inputList(w)
          inputList(w) = inputList(r)
          inputList(r) = tmp
          w = w - 1
        }
        else {
          r = r + 1
        }
      }

      if (inputList(r)._2 > mid)
        r = r - 1

      if (n <= r){
        to = r
      } else {
        from = r + 1
      }
    }

    inputList(n)._2
  }

  def getTopRules(rb: List[(Int, ((Double, Double), List[(Double, Double)]))], antecedent: List[(Double, Double)]): List[Int] = {
    /*getTopKWithQSel(rb.map(r => {
      (r._1.asInstanceOf[Int], getRuleDistance(antecedent, r._2._2))
    }).collect().toList, this.rno).map(_._1)*/
    rb.map(r => (r._1.asInstanceOf[Int], getRuleDistance(r._2._2, antecedent))).sortBy(_._2).map(_._1).take(this.rno).toList
  }

  def persistNearestNeighbors(): RDD[(Int, List[Int])] = {
    testData.zipWithIndex().map{case (k, v) => (v, k)}
      .map(r => (r._1.asInstanceOf[Int], getSortedNeighbors(r._2.features)))
      //.saveAsTextFile("neighbors")
    //this.fullData.zipWithIndex().map{case (k, v) => (v, k)}.map(r => (r._1.asInstanceOf[Int], getSortedNeighbors(r._2.features)))
    //  .saveAsTextFile("neighbors.txt")
  }

  def getSortedNeighbors(t: Vector): List[Int] = {
    this.dataWithIndex.map(r => (r._1.asInstanceOf[Int], getDistance(t, r._2.features))).sortBy(_._2).map(_._1).collect().toList
  }

  def getRuleHashBits(point: List[(Double, Double)], hps: List[List[(Double,Double)]]): SparseVector = {
    val pal = new DenseVector(hps.map(r => {
          val dist = getRuleDistance(point, r)
          //println("---------------------------------------------------------------------" +dist)
          if (dist < ruleDistThresh)
            0.0
          else
            1.0
         }
        ).toArray)
    //println(pal.toString())
    pal.toSparse
  }  
  
  def getHashBitsString(point: LabeledPoint, hps: List[List[Double]]): String = {
    hps.map(r => { 
          val dist = getDistance(point.features.toDense, new DenseVector(r.toArray) )
          //println(dist)
          if (dist < distThresh)
            0
          else
            1
         }
        ).mkString("")
  }
  

  
  def getHashBits(point: LabeledPoint, hps: List[List[Double]]): SparseVector = {
    val pal = new DenseVector(hps.map(r => {
          val dist = getDistance(point.features.toDense, new DenseVector(r.toArray) )
          //println(dist)
          if (dist < distThresh)
            0.0
          else
            1.0
         }
        ).toArray)
    //println(pal.toString())
    pal.toSparse
  }  
  
  def generateRandomRuleHyperPlanes(): List[List[(Double, Double)]] = {
    (1 to hpNo).toList.map { x => 
      val tmp = ruleUniqs.map(r =>         
          r.keySet.toList(scala.util.Random.nextInt(r.keySet.size))
        )
        tmp
      }
  }
  
  def generateRandomHyperPlanes(): List[List[Double]] = {
    (1 to hpNo).toList.map { x => 
      val tmp = uniqs.map(r =>         
          r.keySet.toList(scala.util.Random.nextInt(r.keySet.size))
        )
        tmp
      }
  }
  
  def generateSmartHyperPlanes(): List[List[Double]] = {
    //for now just calls the random version. Later on it can be improved.
    generateRandomHyperPlanes()
  }  
  
  def getTopNeighborsForRuleGeneration(caseIndex: Int): List[Int] = {
    var result : List[(Int, Double)] = List()
    for (i <- 0 until dataWithIndexList.size){
      if (caseIndex != i){
        result = result ::: List((i, getDistance(this.dataWithIndexList(caseIndex)._2.features, this.dataWithIndexList(i)._2.features)))
      }
    }
    //this.dataWithIndex.collect().foreach(r => {
    //  result = result ::: List((r._1.asInstanceOf[Int], getDistance(this.dataWithIndex.lookup(caseIndex)(0).features, r._2.features)))
    //})
    getTopKWithQSel(result, this.ruleRadius).map(_._1)
  }

  def getTopNeighbors(t:Vector): List[Int] = {
    getTopKWithQSel(this.dataWithIndex.map(r => {
      (r._1.asInstanceOf[Int], getDistance(t, r._2.features))
    }).collect().toList, this.k).map(_._1)
    //sortBy(_._2).map(_._1).take(this.k).toList
    /*var result = List[(Int, Double)]()
    this.dataWithIndex.foreach(r => {
      val tempDist = getDistance(t, r._2.features)
      //println(tempDist)
      result = result ::: List((r._1.asInstanceOf[Int], tempDist))
    })
    result.sortBy(_._2).map(_._1).take(this.k)*/
    //this.dataWithIndex.map(r => (r._1.asInstanceOf[Int], getDistance(t, r._2.features))).sortBy(_._2).map(_._1).take(this.k).toList
    //val tmp = this.dataWithIndex.map(r => (r._1.asInstanceOf[Int], getDistance(t, r._2.features))).sortBy(_._2).collect().toList
    //tmp.filter(_._2 <= tmp(this.k)._2).map(_._1)
  }

  def predictKNN(testData: Vector): Double = {
    //kNN
    getTopNeighbors(testData).map(dataWithIndex.lookup(_)(0).label).groupBy(identity).maxBy(_._2.size)._1
    //EAC
    /*val baseCaseIndices = getTopNeighbors(testData)
    var result = 0.0
    baseCaseIndices.map(r => {
      val baseLabel = dataWithIndex.lookup(r)(0).label
      val antecedent = dataWithIndex.lookup(r)(0).features.toArray.zip(testData.toArray).toList
      val rulesToConsider = ruleBase4WithIndex.filter{case (a, b) => b._1._1 == baseLabel}
      getTopRules(rulesToConsider, antecedent).map(ruleBase4WithIndex.lookup(_)(0)._1._2).groupBy(identity).maxBy(_._2.size)._1
    }).groupBy(identity).maxBy(_._2.size)._1*/
  }

  def predictLsh(testData: Vector): Double = {
    0.0
//    //kNN
//    //getTopNeighbors(testData).map(dataWithIndex.lookup(_)(0).label).groupBy(identity).maxBy(_._2.size)._1
//    //EAC
//    
//    val baseCaseIndices = annModel.neighbors(testHashedDataset, this.k).first()._2.map(r => r._1.toInt) 
//    var result = 0.0
//    //println(this.ruleBase4WithIndex.toString())
//    //System.exit(0)
//    
//    baseCaseIndices.map(r => {
//      val baseLabel = dataWithIndexList(r)._2.label
//      val antecedent = dataWithIndexList(r)._2.features.toArray.zip(testData.toArray).toList
//      val rulesToConsider = ruleBase4RddIndex.filter{case (a, b) => b._1._1 == baseLabel}
//      val ttt = getTopRules(rulesToConsider, antecedent).map(ruleBase4WithIndex(_)._2._1._2).groupBy(identity).maxBy(_._2.size)._1
//      //println(ttt.toString)
//      //System.exit(0)
//      ttt
//    }).groupBy(identity).maxBy(_._2.size)._1
//
//    /*baseCaseIndices.map(r => {
//      val baseLabel = dataWithIndex.lookup(r)(0).label
//      val antecedent = dataWithIndex.lookup(r)(0).features.toArray.zip(testData.toArray).toList
//      val rulesToConsider = ruleBase4WithIndex.filter{case (a, b) => b._1._1 == baseLabel}
//      getTopRules(rulesToConsider, antecedent).map(ruleBase4WithIndex.lookup(_)(0)._1._2).groupBy(identity).maxBy(_._2.size)._1
//    }).groupBy(identity).maxBy(_._2.size)._1*/
  }
  
  def predict(testData: Vector): Double = {
    //kNN
    //getTopNeighbors(testData).map(dataWithIndex.lookup(_)(0).label).groupBy(identity).maxBy(_._2.size)._1
    //EAC
    
    val baseCaseIndices = if (this.useLsh) annModel.neighbors(testHashedDataset, this.k).first()._2.map(r => r._1.toInt).toList else getTopNeighbors(testData) 
    var result = 0.0
    //println(this.ruleBase4WithIndex.toString())
    //System.exit(0)

    baseCaseIndices.map(r => {
      val baseLabel = dataWithIndexList(r)._2.label
      val antecedent = dataWithIndexList(r)._2.features.toArray.zip(testData.toArray).toList
      val rulesToConsider = ruleBase4WithIndex.filter{case (a, b) => b._1._1 == baseLabel}
      val ttt = getTopRules(rulesToConsider, antecedent).map(ruleBase4WithIndex(_)._2._1._2).groupBy(identity).maxBy(_._2.size)._1
      //println(ttt.toString)
      //System.exit(0)
      ttt
    }).groupBy(identity).maxBy(_._2.size)._1

    /*baseCaseIndices.map(r => {
      val baseLabel = dataWithIndex.lookup(r)(0).label
      val antecedent = dataWithIndex.lookup(r)(0).features.toArray.zip(testData.toArray).toList
      val rulesToConsider = ruleBase4WithIndex.filter{case (a, b) => b._1._1 == baseLabel}
      getTopRules(rulesToConsider, antecedent).map(ruleBase4WithIndex.lookup(_)(0)._1._2).groupBy(identity).maxBy(_._2.size)._1
    }).groupBy(identity).maxBy(_._2.size)._1*/
  }

  def train(sc: SparkContext): EACModel = {
    val classStat = data.map(x => x.label).countByValue()
    
    var featureStat = List[Map[Double, Long]]()
    var featureClassStat = List[Map[(Double, Double), Long]]()
    for (i <- 0 until data.first().features.size){
      if (categoricalFeaturesInfo.keySet.contains(i)) {
        val tmp = data.map(x => x.features(i)).countByValue()
        featureStat =  featureStat ::: List(tmp)
        val tmp2 = data.map(x => (x.features(i), x.label)).countByValue()
        featureClassStat = featureClassStat ::: List(tmp2)
      }
      else
      {
        featureStat =  featureStat ::: List(Map[Double,Long]())
        featureClassStat = featureClassStat ::: List(Map[(Double, Double), Long]())
      }
    }
//    val classStatb = sc.broadcast(classStat)
//    val featureStatb = sc.broadcast(featureStat)
//    val featureClassStatb = sc.broadcast(featureClassStat)

    


//    println("########################################################" + annModel.neighbors(hashedDataset.filter(r => r._1 == 0), this.ruleRadius + 1).first()._2.map(r => r._1.toInt).toList.toString())        
//    println("********************************************************" + this.ruleRadius + "*********" + annModel.neighbors(hashedDataset.filter(r => r._1 == 0), this.ruleRadius + 1).first()._2.map(r => r._1.toInt).toList.filter { x => x != 0 })
//    System.exit(0)        
//    val zer = annModel.neighbors(hashedDataset.filter(r => r._1 == 0), 100)
//    
//    zer.collect().foreach(f => println(f._1.toString()))
//    zer.collect().foreach(f => f._2.foreach(m => println("============================================================" + m.toString()) ))
    
    //hashedDataset.take(100).foreach(r => println(r.toString()))
    //println(featureStat.toString())
//	System.exit(0)
    //println(featureClassStat.toString())
    /*this.dataWithIndex = data.zipWithIndex().map{case (k,v) => (v, k)}
    //key: class value, value: how many records have this class value
    var classStat = scala.collection.mutable.Map[Double, Int]()
    //each element in the list is a hashmap with key: feature value, value: how many record have that value for the corresponding feature
    var featureStat = List.fill(data.first().features.size)(scala.collection.mutable.Map[Double, Int]())
    //each element in the list is a hashmap with key: tuple of feature value and class value, value: how many records match the key
    var featureClassStat = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double), Int]())
    var test = 0
    mizan = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double), Double]())
    println("=================================STARTED building featureStat and classStat=======================")
    data.foreach(r => {
      //featureStat.add(new util.HashMap[Double, Int])
      //featureClassStat.add(new util.HashMap[(Double, Double), Int])
      println(classStat.toString())
      println(test)
      test += 1

      var vectorIndex = 0
      if (classStat.contains(r.label)) {
        classStat(r.label) = classStat(r.label) + 1
      }
      else {
        classStat(r.label) = 1
      }
      r.features.toArray.foreach(f => {
        if (featureStat(vectorIndex).contains(f))
          featureStat(vectorIndex)(f) = featureStat(vectorIndex)(f) + 1
        else
          featureStat(vectorIndex)(f) = 1
        if (featureClassStat(vectorIndex).contains((f, r.label)))
          featureClassStat(vectorIndex)((f, r.label)) =  featureClassStat(vectorIndex)((f, r.label)) + 1
        else
          featureClassStat(vectorIndex)((f, r.label)) = 1
        vectorIndex += 1
      })
    })*/
    println("=================================STARTED building mizan=======================")
    //println(classStat.toString())
    //featureClassStat(0).keys.foreach(println(_))
    //var mizan2 = List.fill(data.first().features.size)(scala.collection.mutable.Map[(Double, Double),  Double]())
    val featureIt = featureStat.iterator
    var featureCounter = 0
    //the following while loop generates VDM between all possible pairs of values for all features in the domain
    while(featureIt.hasNext){
      //println("feature iterator")
      val featureValues = featureIt.next.keySet.toArray
      if (categoricalFeaturesInfo.keySet.contains(featureCounter)) {
        for (i <- 0 until featureValues.length) {
          for (j <- i + 1 until featureValues.length) {
            val v1 = featureValues(i).asInstanceOf[Double]
            val v2 = featureValues(j).asInstanceOf[Double]
            val v1cnt = featureStat(featureCounter)(v1).toInt.toDouble
            val v2cnt = featureStat(featureCounter)(v2).toInt.toDouble
            var vdm = 0.0
            val classValsIt = classStat.keySet.iterator
            while (classValsIt.hasNext) {
              val classVal = classValsIt.next()
              val tmp1 = featureClassStat(featureCounter).getOrElse(((v1, classVal)), 0L).toInt.toDouble
              val tmp2 = featureClassStat(featureCounter).getOrElse(((v2, classVal)), 0L).toInt.toDouble
              vdm += Math.abs(tmp1 / v1cnt - tmp2 / v2cnt)
              //println(tmp1 + " " + tmp2 + " " + " " + tmp1 + " " +tmp2 +" "+ vdm)
            }
            //I'll put the smaller element as the first element of the tuple.
            //this makes looking up a tuple in mizan easier in future (meaning that if I want to check the
            // distance between two values, I'll always put the smaller value as the first element in the look up as well)

            //println(featureClassStat.toString())
            if (v1 <= v2) {
              mizan(featureCounter)((v1, v2)) = vdm
              //println(vdm)
            }
            else {
              mizan(featureCounter)((v2, v1)) = vdm
              //println(vdm)
            }
          }
        }
      }
      featureCounter += 1
    }
    //this.mizanb = sc.broadcast(mizan)
    
    uniqs = featureStat
    val hyperPlanes:List[List[Double]] = generateRandomHyperPlanes()
    val hashedDataset = dataWithIndex.map(r => {
      //println("test " + r._1.toString() + " " + r._2.toString())
      //println(getHashBits(r._2, hyperPlanes).toString())
      (r._1, getHashBits(r._2, hyperPlanes)) } )
      
    
    this.testHashedDataset = testWithIndex.map(r => (r._1, getHashBits(r._2, hyperPlanes)) )
    //hashedDataset.foreach(f => println(f._1 + " " + f._2.toString()))
    //hashedDataset.foreach(r => println(r._2.toString()))
    //System.exit(0)
    
    this.annModel =
      new com.github.karlhigley.spark.neighbors.ANN(dimensions = hpNo, measure = "jaccard")
        .setTables(4)
        .setSignatureLength(128)
        .setPrimeModulus(739)
        .setBands(16)
        .train(hashedDataset)

//    val zer = annModel.neighbors(hashedDataset.filter(r => r._1 == 0), 100)
//    
//    //zer.collect().foreach(f => println(f._1.toString()))
//    zer.collect().foreach(f => f._2.foreach(m => println("============================================================" + m.toString()) ))
        
    //println(mizan.toString())
	//System.exit(0)
    /*println("=================================STARTED building distances=======================")
    //the following section generates distances between all pairs of cases in the underlying domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        //I'll put the smaller element as the first element of the tuple.
        distances.put((i,j), getDistance(i,j))
      }
    }

    println("=================================STARTED building neighbors=======================")
    //the following section list nearest neighbors for all cases in the domain
    for (i <- 0 until dataWithIndex.count().asInstanceOf[Int]){
      for (j <- i + 1 until dataWithIndex.count().asInstanceOf[Int]){
        neighbors(i) = getNearestNeighbors(i)
      }
    }*/
    //persistNearestNeighbors()
    //null
    //System.exit(0)
    

        
    println("Started forming rules")
//    println("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" + annModel.neighbors(hashedDataset.filter(r => r._1 == 0), this.ruleRadius + 1).first()._2.map(r => r._1.toInt).toList.toString())
//    System.exit(0)
    //val caseNeighbors: scala.collection.mutable.Map[Int, List[Int]] = scala.collection.mutable.Map[Int, List[Int]]()
    
    //val caseNeighbors = dataWithIndex.map(t => (t._1.toInt, annModel.neighbors(hashedDataset.filter(r => r._1 == t._1), this.ruleRadius + 1).first()._2.map(r => r._1.toInt).toList.filter { x => x != t._1 }))
    
//    val caseNeighbors = dataWithIndex.map(t => 
//      (t._1, annModel.neighbors(hashedDataset.filter(r => r._1 == t._1), this.ruleRadius + 1)
//          .first()._2.map(r => r._1).toList.filter { x => x != t._1 }))
          
    val caseNeighbors = annModel.neighbors(hashedDataset, this.ruleRadius + 1)
    .map(f => (f._1, f._2.filter(p => p._1 != f._1).map(r => r._1).toList))
          

    val caseNeighborsVector = caseNeighbors.flatMap(f => f._2.map { x => (x, f._1) })
    .join(dataWithIndex).map(f => (f._2._1, f._2._2)).groupByKey().map(f => (f._1, f._2.toList))
    
    //System.exit(0)
    //println(caseNeighbors.toString())
//    generateRandomHyperPlanes()
//    null
    //System.exit(0)
    //val tmpCb = dataWithIndex.collect().toList
    
    ruleBase4RddIndex = dataWithIndex.join(caseNeighborsVector)
    .flatMap(f => f._2._2.map { x => ((f._2._1.label, x.label),(f._2._1.features.toArray.toList.zip(x.features.toArray.toList))) })
    .zipWithIndex().map{case (k, v) => (v, k)}
    
    val ruleBase4RddIndexReverse = dataWithIndex.join(caseNeighborsVector)
    .flatMap(f => f._2._2.map { x => ((x.label, f._2._1.label),(x.features.toArray.toList.zip(f._2._1.features.toArray.toList))) })
    .zipWithIndex().map{case (k, v) => (v, k)}
    
    ruleBase4RddIndex.union(ruleBase4RddIndexReverse)


//    ruleBase4 = dataWithIndexList.map(r => (r, caseNeighbors(r._1.asInstanceOf[Int]))).map{case (k,v) => v.map(p => {
//      ((k._2.label, dataWithIndexList(p)._2.label), (k._2.features.toArray.toList.zip(dataWithIndexList(p)._2.features.toArray)))
//    })}.flatMap(q => q)
//
//    /*ruleBase4 = dataWithIndex.map(r => (r, caseNeighbors(r._1.asInstanceOf[Int]))).map{case (k, v) =>  v.map(p => {
//      val tmpCase = dataWithIndexList(p)._2
//      ((k._2.label, tmpCase.label), (k._2.features.toArray.toList.zip(tmpCase.features.toArray)))
//    })}.flatMap(q => q)*/
//
//    val tmpRuleBase = dataWithIndexList.map(r => (r, caseNeighbors(r._1.asInstanceOf[Int]))).map{case (k, v) =>  v.map(p => {
//      val tmpCase = dataWithIndexList(p)._2
//      ((tmpCase.label, k._2.label), (tmpCase.features.toArray.toList.zip(k._2.features.toArray)))
//    })}.flatMap(q => q)
//
//    ruleBase4.union(tmpRuleBase)
//    ruleBase4WithIndex = ruleBase4.zipWithIndex.map{case (k,v) => (v,k)}
    //ruleBase4WithIndex = ruleBase4.zipWithIndex(). .map{case (k, v) => (v, k)}

      //cartesian(dataWithIndex).filter{case (a,b) => a._1 != b._1}
      //.map{case ((a,b),(c,d)) => ((b.label, d.label), (b.features.toArray.toList zip d.features.toArray.toList))}
    //var ruleBase = dataWithIndex.cartesian(dataWithIndex).filter{case (a,b) => a._1 != b._1}
    //  .map{case ((a,b),(c,d)) => ((b.label, d.label), (b.features.toArray.toList zip d.features.toArray.toList))}
//    val ruleClassStat = ruleBase4.map(x => x._1).groupBy(identity).mapValues(_.size)
    val ruleClassStat = ruleBase4RddIndex.map(x => x._2._1).groupBy(identity).mapValues(_.size).collect().toMap
    

    var ruleFeatureStat = List[Map[(Double, Double), Int]]()
    var ruleFeatureClassStat = List[Map[((Double, Double), (Double, Double)), Int]]()
    for (i <- 0 until data.first().features.size){
      if (categoricalFeaturesInfo.keySet.contains(i)) {
//        val tmp = ruleBase4.map(x => x._2(i)).groupBy(identity).mapValues(_.size)
        val tmp = ruleBase4RddIndex.map(x => x._2._2(i)).groupBy(identity).mapValues(_.size).collect().toMap
        ruleFeatureStat = ruleFeatureStat ::: List(tmp)
//        val tmp2 = ruleBase4.map(x => (x._2(i), x._1)).groupBy(identity).mapValues(_.size)
        val tmp2 = ruleBase4RddIndex.map(x => (x._2._2(i), x._2._1)).groupBy(identity).mapValues(_.size).collect().toMap
        ruleFeatureClassStat = ruleFeatureClassStat ::: List(tmp2)
      }
      else{
        ruleFeatureStat = ruleFeatureStat ::: List()
        ruleFeatureClassStat = ruleFeatureClassStat ::: List()
      }
    }

    ruleUniqs = ruleFeatureStat
    this.ruleHyperPlanes = generateRandomRuleHyperPlanes()
//    val hashedRuleset = ruleBase4RddIndex.map(r => {
//      //println("test " + r._1.toString() + " " + r._2.toString())
//      //println(getHashBits(r._2, hyperPlanes).toString())
//      (r._1, getRuleHashBits(r._2._2, ruleHyperPlanes)) } )
//      
//    
//    //this.testHashedDataset = testWithIndex.map(r => (r._1, getHashBits(r._2, hyperPlanes)) )
//    //hashedDataset.foreach(f => println(f._1 + " " + f._2.toString()))
//    //hashedRuleset.foreach(r => println(r._2.toString()))
//    //println("()((((((((((((((((((((()))))))))))))))))))))" + hashedRuleset.count() + "---" + hashedRuleset.filter(r => r._2.indices.length == 0).count())
//    
//    //System.exit(0)
//    
//    this.annRuleModel =
//      new com.github.karlhigley.spark.neighbors.ANN(dimensions = 1000, measure = "jaccard")
//        .setTables(4)
//        .setSignatureLength(128)
//        .setPrimeModulus(739)
//        .setBands(16)
//        .train(hashedRuleset)
        
        

    val ruleFeatureIt = ruleFeatureStat.iterator
    var ruleFeatureCounter = 0
    //the following while loop generates VDM between all possible pairs of values for all features in the domain
    while(ruleFeatureIt.hasNext){
      //println("feature iterator")
      val ruleFeatureValues = ruleFeatureIt.next.keySet.toArray
      if (categoricalFeaturesInfo.keySet.contains(ruleFeatureCounter)) {
        for (i <- 0 until ruleFeatureValues.length) {
          for (j <- i + 1 until ruleFeatureValues.length) {
            val v1 = ruleFeatureValues(i)
            val v2 = ruleFeatureValues(j)
            val v1cnt = ruleFeatureStat(ruleFeatureCounter)(v1).toInt.toDouble
            val v2cnt = ruleFeatureStat(ruleFeatureCounter)(v2).toInt.toDouble
            var vdm = 0.0
            val ruleClassValsIt = ruleClassStat.keySet.iterator
            while (ruleClassValsIt.hasNext) {
              val ruleClassVal = ruleClassValsIt.next()
              val tmp1 = ruleFeatureClassStat(ruleFeatureCounter).getOrElse(((v1, ruleClassVal)), 0).toDouble
              val tmp2 = ruleFeatureClassStat(ruleFeatureCounter).getOrElse(((v2, ruleClassVal)), 0).toDouble
              vdm += Math.abs(tmp1 / v1cnt - tmp2 / v2cnt)
              //println(tmp1 + " " + tmp2 + " " + " " + tmp1 + " " +tmp2 +" "+ vdm)
            }
            //I'll put the smaller element as the first element of the tuple.
            //this makes looking up a tuple in mizan easier in future (meaning that if I want to check the
            // distance between two values, I'll always put the smaller value as the first element in the look up as well)

            //println(featureClassStat.toString())
            //println("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            ruleMizan(ruleFeatureCounter)((v1, v2)) = vdm
            //ruleMizan(featureCounter)((v2,v1)) = vdm
            //if (v1 <= v2) {
            //  ruleMizan(featureCounter)((v1, v2)) = vdm
            //println(vdm)
            //}
            //else {
            //  ruleMizan(featureCounter)((v2, v1)) = vdm
            //println(vdm)
            //}
          }
        }
      }
      ruleFeatureCounter += 1
    }

	//println("*********************((((((((((((((((((((((((()))))))))))))))))))))))))" +ruleMizan.toString)
	//System.exit(0)
    //ruleMizan(0).count()
    //println(ruleMizan.toString())
    //System.exit(0)
    //println("Started building rule mizan")
    println("IS ABOUT TO BUILD THE MODEL")
    new EACModel(k)
  }
}

/*object EAC {
  def train(input: RDD[LabeledPoint]): EACModel = {
    new EAC(0, input).run()
  }
}*/

class EACModel (inputK: Int)
  extends ClassificationModel with Serializable with Saveable{
  private val k = inputK



  override def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.map(t => {
      predict(t)
    })
  }

  override def predict(testData: Vector): Double = {
    0.0
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    println("test")
  }
}
