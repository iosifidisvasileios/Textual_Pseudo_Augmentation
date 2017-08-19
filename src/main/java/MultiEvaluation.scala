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
  val word2vecFromTwitterMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromGoogleSentimentMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromTwitterSentimentMapper = new mutable.HashMap[String,String]()  { }
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

  def distant_supervision_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: Array[RDD[LabeledPoint]],
                                     data: RDD[String], htf: HashingTF, sc: SparkContext) : Unit = {
    //------------------------------------------ Balanced Scenarios ------------------------------------------------------------
    println("Simple Balanced Scenario                                ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0), dataSplit(1)))

    for(i <- 0 to 5) {
      println("Balanced Scenario with Antonymous                       ---------------------------------------------------")
      callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0)
        .union(syntheticGenerationFunctions.choose_antonymous_generation_function(data, htf, wordnetMapper, i)), dataSplit(1)))
    }

    println("Balanced Synonymous with Google                         ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0)
      .union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleMapper)), dataSplit(1)))

    println("Balanced Synonymous with WORD2VEC Google Sentiment      ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0)
      .union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleSentimentMapper)), dataSplit(1)))

    println("Balanced Synonymous with WORD2VEC Twitter Sentiment     ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.simple10fold(dataSplit(0)
      .union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(data, htf, word2vecFromTwitterSentimentMapper)), dataSplit(1)))

    println("Balanced Scenario with BLANKOUT                         ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.balanced10foldWithBlankout(data, dataSplit(1), htf))

    //---------------------------------------------- Imbalance & Over/under-Sampling -----------------------------------

    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10fold(dataSplit(0), dataSplit(1)))

    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithOversampling(data, dataSplit(1), htf))

    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithUndersampling(data, dataSplit(1), htf, sc))

    //------------------------------------------BLANKOUT Imbalance----------------------------------------------------
    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation-----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinority(data, dataSplit(1), htf))

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation   -----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(data, dataSplit(1), htf, sc))

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation   -----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(data, dataSplit(1), htf, sc))


    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation-----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClasses(data, dataSplit(1), htf))

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation   -----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSampling(data, dataSplit(1), htf, sc))

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation   -----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSampling(data, dataSplit(1), htf, sc))

    //------------------------------------------ANTONYMOUS Imbalance----------------------------------------------------
    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation         -----------------------------------")
      callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonyms(
        data, dataSplit(1), htf, wordnetMapper, i))

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation----------")
      callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSampling(
        data, dataSplit(1), htf, wordnetMapper, i, sc))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation-----------")
      callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSampling(
        data, dataSplit(1), htf, wordnetMapper, i, sc))
    }

    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(
      data, dataSplit(1), htf, wordnetMapper, 0, sc))

    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(
      data, dataSplit(1), htf, wordnetMapper, 2, sc))

    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(
      data, dataSplit(1), htf, wordnetMapper, 4, sc))

    //-------------------------------------------------------------------------------------------------------------------
    //-------------------------- WORD2VEC NO-SENTIMENT EMBEDDINGS ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES  --------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, dataSplit(1), htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES -----------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, dataSplit(1), htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromGoogleMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromGoogleMapper,sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromGoogleMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromGoogleMapper, sc))

    //----------------------------------------------------- TWITTER NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES  --------------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, dataSplit(1), htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES -----------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, dataSplit(1), htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromTwitterMapper,sc))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromTwitterMapper,sc))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromTwitterMapper, sc))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromTwitterMapper, sc))

    //-------------------------------------------------------------------------------------------------------------------
    //------------------------ WORD2VEC SENTIMENT EMBEDDINGS  ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Google- Sentiment----------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Google- Sentiment------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS Google- Sentiment -------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Google- Sentiment --------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -----")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Google- Sentiment -----")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Google- Sentiment --")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromGoogleSentimentMapper, sc))

    //----------------------------------------------------- TWITTER SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Twitter- Sentiment---------------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Twitter- Sentiment-----------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  Twitter- Sentiment------------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Twitter- Sentiment--------------------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment-----")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  Twitter- Sentiment------")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING Twitter- Sentiment--")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING Twitter- Sentiment---")
    callBinaryMetricsFunction(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, dataSplit(1), htf, word2vecFromTwitterSentimentMapper, sc))

  }

  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName(this.getClass.getName)
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // load antonymous dictionary from local directory
    for (line <- Source.fromFile("wordnet_antonymous.txt").getLines) {
      val words =line.split(",")
      wordnetMapper += (words(0).toLowerCase -> words(1).toLowerCase)
    }

    // load similar dictionary from local directory (google w2vec)
    for (line <- Source.fromFile("similar_words.txt").getLines) {
      val words = line.split("\t")
      word2vecFromGoogleMapper += (words(0).toLowerCase -> words(1).split(",")(0).toLowerCase)
    }    // load similar dictionary from local directory (google w2vec)


    for (line <- Source.fromFile("similar_words_based_on_twitter.txt").getLines) {
      val words = line.split("\t")
      word2vecFromTwitterMapper += (words(0).toLowerCase -> words(1).split(",")(0).toLowerCase)
    }


    for (line <- Source.fromFile("similar_words_sentiment_voc.txt").getLines) {
      val words = line.split("\t")
      word2vecFromGoogleSentimentMapper += (words(0).toLowerCase -> words(1).split(",")(0).toLowerCase)
    }

    for (line <- Source.fromFile("similar_words_twitter_embeddings_with_sentiment.txt").getLines) {
      val words = line.split("\t")
      word2vecFromTwitterSentimentMapper += (words(0).toLowerCase -> words(1).split(",")(0).toLowerCase)
    }

    // load similar dictionary from local directory (dist. sup. w2vec)
    for (line <- Source.fromFile("similar_words_based_on_distant_supervision_trained_model.txt").getLines) {
      val words = line.split("\t")
      word2vecFromDatasetMapper += (words(0).toLowerCase -> words(1).split(",")(0).toLowerCase)
    }

    // read input from HDFS
    val data = sc.textFile(args(0))
    val semEvalData = sc.textFile(args(1))
    val tsentiment15Data = sc.textFile(args(2))
    val options = args(3).toInt

    // assign words to numbers converter
    val htf = new HashingTF(1500000)
    val htf_2 = new HashingTF(1500000)

    options match {
      case 1 =>
        //----------------------Sentiment140-------------------------------
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
        //        println("--------- Testing set count ---------")
        //        println("test set positive counts : " + dataSplit(1).filter(_.label == 0 ).count())
        //        println("test set negative counts : " + dataSplit(1).filter(_.label == 1 ).count())
        //        println("-------------------------------------")
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSplit, data, htf, sc)

      case 2 =>
        //-----------------------SemEval----------------------------------

        val finalsemEvalData = semEvalData.filter{line=>
          val parts = line.split(',')
          parts(1).equals("negative")
        }.union(sc.parallelize(semEvalData.filter{line=>
          val parts = line.split(',')
          parts(1).equals("positive")
        }.take(11427)))

        // load dataset to RDD
        val dataSetSemEval = finalsemEvalData.map { line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf_2.transform(parts(2).split(' ')))}.cache()
        //0 for positive 1 for negative
        //        println("positive: " + dataSetSemEval.filter(_.label == 0.0).count)
        //        println("negative: " + dataSetSemEval.filter(_.label == 1.0).count)

        // force balance into the dataset by dropping positive instances
//        val temp_balanced = dataSetSemEval.filter(_.label == 1.0).union(sc.parallelize(dataSetSemEval.filter(_.label == 0.0).take(11427)))

        // split dataset for hold out eval
        val dataSplitSemEval = dataSetSemEval.randomSplit(Array(0.7, 0.3))
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSplitSemEval, finalsemEvalData, htf, sc)

      case 3 =>
        //-----------------------TSentiment15----------------------------------

        val finalTSentiment15 = tsentiment15Data.filter{line=>
          val parts = line.split(',')
          parts(1).equals("negative")
        }.union(sc.parallelize(tsentiment15Data.filter{line=>
          val parts = line.split(',')
          parts(1).equals("positive")
        }.take(316663)))

        // load dataset to RDD
        val dataSetTSentiment15 = finalTSentiment15.map { line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf_2.transform(parts(2).split(' ')))}.cache()
        val dataSplitTSentiment15 = dataSetTSentiment15.randomSplit(Array(0.7, 0.3))
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSplitTSentiment15, finalTSentiment15, htf, sc)

    }

  }
}

