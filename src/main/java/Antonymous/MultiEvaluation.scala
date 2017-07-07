package Antonymous

/**
  * Created by iosifidis on 19.09.16.
  */

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object MultiEvaluation {

  val syntheticGenerationFunctions = new SyntheticGenerationFunctions
  val word2vecFromGoogleMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromDatasetMapper = new mutable.HashMap[String,String]()  { }
  val wordnetMapper  = new mutable.HashMap[String,String]()  { }

  def callBinaryMetricsFunction(predictionAndLabels: RDD[(Double, Double)]): BinaryClassificationMetrics ={
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val precision = metrics.precisionByThreshold.collectAsMap()
    println(" Precision for class positive: " + precision.get(0.0))
    println(" Precision for class negative: " + precision.get(1.0))
    val recall = metrics.recallByThreshold().collectAsMap()
    println(" Recall for class positive: " + recall.get(0.0))
    println(" Recall for class negative: " + recall.get(1.0))
    val PRC = metrics.pr
    val f1Score = metrics.fMeasureByThreshold.collectAsMap()
    println(" f1Score for class positive: " + f1Score.get(0.0))
    println(" f1Score for class negative: " + f1Score.get(1.0))
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
    metrics
  }




  def sem_eval_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplitSemEval: Array[RDD[LabeledPoint]], semEvalData: RDD[String], htf_2: HashingTF, sc:SparkContext): Unit = {
    println("Simple Balanced Scenario SemEval      ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplitSemEval(0), dataSplitSemEval(1)))

    println("Balanced Scenario with Antonymous   ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplitSemEval(0).union(syntheticGenerationFunctions.antonymsGeneratorSemEval(semEvalData, htf_2, wordnetMapper)), dataSplitSemEval(1)))

    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10fold(dataSplitSemEval(0), dataSplitSemEval(1)))

    println("Imbalanced Scenario with Antonymous ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsSemEval(semEvalData, dataSplitSemEval(1), htf_2, wordnetMapper))

    println("Imbalanced Scenario with OverSampling Sem EVAL  ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithOversamplingSemEval(semEvalData, dataSplitSemEval(1), htf_2))

    println("Imbalanced Scenario with UnderSampling Sem EVAL ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithUndersamplingSemEval(semEvalData, dataSplitSemEval(1), htf_2, sc))


    println("Balanced Synonymous with Google     ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplitSemEval(0).union(syntheticGenerationFunctions.similarGeneratorSemEval(semEvalData, htf_2, word2vecFromGoogleMapper)), dataSplitSemEval(1)))
    println("Balanced Synonymous with Dataset    ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplitSemEval(0).union(syntheticGenerationFunctions.similarGeneratorSemEval(semEvalData, htf_2, word2vecFromDatasetMapper)), dataSplitSemEval(1)))

    println("Imbalanced Synonymous with Google   ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsSemEval(semEvalData, dataSplitSemEval(1), htf_2, word2vecFromGoogleMapper))
    println("Imbalanced Synonymous with Dataset  ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsSemEval(semEvalData, dataSplitSemEval(1), htf_2, word2vecFromDatasetMapper))

  }

  def distant_supervision_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: Array[RDD[LabeledPoint]], data: RDD[String], htf: HashingTF, sc: SparkContext) : Unit = {
//    println("Simple Balanced Scenario            ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0), dataSplit(1)))
//    println("Balanced Scenario with Antonymous   ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0).union(syntheticGenerationFunctions.antonymsGenerator(data, htf, wordnetMapper)), dataSplit(1)))
//
//    println("Simple Imbalanced Scenario          ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10fold(dataSplit(0), dataSplit(1)))

    println("Imbalanced Scenario with Antonymous ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonyms(data, dataSplit(1), htf, wordnetMapper))
//
//    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithOversampling(data, dataSplit(1), htf))
//
//    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithUndersampling(data, dataSplit(1), htf, sc))
//
//    println("Balanced Synonymous with Google     ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0).union(syntheticGenerationFunctions.similarGenerator(data, htf, word2vecFromGoogleMapper)), dataSplit(1)))
//
//    println("Imbalanced Synonymous with Google   ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilars(data, dataSplit(1), htf, word2vecFromGoogleMapper))
//
//    println("Balanced Synonymous with Dataset    ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0).union(syntheticGenerationFunctions.similarGenerator(data, htf, word2vecFromDatasetMapper)), dataSplit(1)))
//
//    println("Imbalanced Synonymous with Dataset  ---------------------------------------------------")
//    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilars(data, dataSplit(1), htf, word2vecFromDatasetMapper))
  }

  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName(this.getClass.getName)
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // load antonymous dictionary from local directory
    for (line <- Source.fromFile("wordnet_antonymous.txt").getLines) {
      val words =line.split(",")
      wordnetMapper += (words(0) -> words(1))
    }

    // load similar dictionary from local directory (google w2vec)
    for (line <- Source.fromFile("similar_words.txt").getLines) {
      val words = line.split("\t")
      word2vecFromGoogleMapper += (words(0) -> words(1).split(",")(0))
    }

    // load similar dictionary from local directory (dist. sup. w2vec)
    for (line <- Source.fromFile("similar_words_based_on_distant_supervision_trained_model.txt").getLines) {
      val words = line.split("\t")
      word2vecFromDatasetMapper += (words(0) -> words(1).split(",")(0))
    }

    // read input from HDFS
    val data = sc.textFile(args(0))
    val semEvalData = sc.textFile(args(1))
    val options = args(2).toInt

    // assign words to numbers converter
    val htf = new HashingTF(1500000)
    val htf_2 = new HashingTF(1500000)

    options match {
      case 1 =>
        // load dataset to RDD
        val dataSet = data.map { line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf.transform(parts(2).split(' ')))}.cache()

        // split dataset for hold out eval
        val dataSplit = dataSet.randomSplit(Array(0.7, 0.3))
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSplit, data, htf, sc)

      case 2 =>
        //-----------------------SemEval----------------------------------
        // load dataset to RDD
        val dataSetSemEval = semEvalData.map { line =>
          val parts = line.split('\t')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf_2.transform(parts(3).split(' ')))}.cache()
        //0 for positive 1 for negative
        //        println("positive: " + dataSetSemEval.filter(_.label == 0.0).count)
        //        println("negative: " + dataSetSemEval.filter(_.label == 1.0).count)

        // force balance into the dataset by dropping positive instances
        val temp_balanced = dataSetSemEval.filter(_.label == 1.0).union(sc.parallelize(dataSetSemEval.filter(_.label == 0.0).take(11427)))

        // split dataset for hold out eval
        val dataSplitSemEval = temp_balanced.randomSplit(Array(0.7, 0.3))
        sem_eval_evaluation(syntheticGenerationFunctions, dataSplitSemEval, semEvalData, htf_2, sc)

      case _ =>
        val dataSet = data.map { line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf.transform(parts(2).split(' ')))}.cache()
        val dataSplit = dataSet.randomSplit(Array(0.7, 0.3))
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSplit, data, htf, sc)

        val dataSetSemEval = semEvalData.map { line =>
          val parts = line.split('\t')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf_2.transform(parts(3).split(' ')))}.cache()

        println("positive: " + dataSetSemEval.filter(_.label == 0.0).count)
        println("negative: " + dataSetSemEval.filter(_.label == 1.0).count)

        // force balance into the dataset by dropping positive instances
        val temp_balanced = dataSetSemEval.filter(_.label == 1.0).union(sc.parallelize(dataSetSemEval.filter(_.label == 0.0).take(11427)))

        // split dataset for hold out eval
        val dataSplitSemEval = temp_balanced.randomSplit(Array(0.7, 0.3))
        sem_eval_evaluation(syntheticGenerationFunctions, dataSplitSemEval, semEvalData, htf_2, sc)
    }

  }
}

