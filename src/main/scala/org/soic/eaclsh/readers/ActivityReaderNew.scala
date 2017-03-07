package org.soic.eaclsh.readers
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import scala.util.Try

object ActivityReaderNew {
  
  val delimiter = ","
  val schemaDelimiter = " "
  
  def readData(sc: SparkContext, inputPath:String, featureFields:String): RDD[LabeledPoint] = {

    val featuresIndexes = dataSchema.split(schemaDelimiter).zipWithIndex.filter{case (value, index) => featureFields.split(schemaDelimiter).contains(value)}.map{case (value, index) => index}.toSet
    
    sc.textFile(inputPath).map(s=>s.split(delimiter).map { x => x.toDouble })
    .map(
        s=>
					{
					  val label = s(6)
				    LabeledPoint(label, Vectors.dense( s.zipWithIndex.filter{case (value, index) => featuresIndexes.contains(index) }.map{case (value, index) => value} ))
					}
    )
  }.cache()
  
  def numberOfClasses: Int = 7

  def categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]((3,9),(4,3),(5,6))
  def categoricalFeaturesInfo2: Map[Int, Int] = Map[Int, Int]()

  def numericalFeaturesInfo: Map[Int, Double] = Map[Int, Double]((0,0.4472657),(1,0.449882),(2,0.5141587))
  
  def numericalFeaturesRange: Map[Int, (Double, Double)] = Map[Int, (Double, Double)]((0, (0,1)),(1, (0,1)),(2, (0,1)) )

  def dataSchema: String = "x y z user model device gt"

  def inputFileName: String = "activity/activityCleaned1k.data"  
}