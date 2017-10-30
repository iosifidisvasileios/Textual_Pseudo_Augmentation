import java.util

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
    println(" Precision for class positive: " + precision.get(0.0).last)
    println(" Precision for class negative: " + precision.get(1.0).last)
    val recall = metrics.recallByThreshold().collectAsMap()
    println(" Recall for class positive: " + recall.get(0.0).last)
    println(" Recall for class negative: " + recall.get(1.0).last)
    val PRC = metrics.pr
    val f1Score = metrics.fMeasureByThreshold.collectAsMap()
    println(" f1Score for class positive: " + f1Score.get(0.0).last)
    println(" f1Score for class negative: " + f1Score.get(1.0).last)
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
    metrics
  }

  def print_results(test_map: mutable.Map[String, Double]) = {
    println("precision_negative:: " + test_map("precision_negative"))
    println("precision_positive:: " + test_map("precision_positive"))

    println("recall_positive:: " + test_map("recall_positive"))
    println("recall_negative:: " + test_map("recall_negative"))
    println("f1_positive:: " + test_map("f1_positive"))
    println("f1_negative:: " + test_map("f1_negative"))

    println("accuracy:: " + test_map("accuracy"))
    println("AucRoc:: " + test_map("AucRoc"))

  }

  def distant_supervision_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: RDD[LabeledPoint],
                                     data: RDD[String], htf: HashingTF, sc: SparkContext, k: Int, sentiwordSet: util.HashSet[String]) : Unit = {
    //    //        ------------------------------------------ Balanced Scenarios ------------------------------------------------------------
    //    println("Simple Balanced Scenario                                ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit))
    //
    //    for(i <- 0 to 5) {
    //      println("Balanced Scenario with Antonymous Scenario = " + i + "  -------------------------------------------------")
    //      print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
    //        .choose_antonymous_generation_function(data, htf, wordnetMapper, i))))
    //
    //    }
    //
    //
    //    println("Balanced Synonymous with Google                         ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleMapper))))
    //
    //
    //    println("Balanced Synonymous with WORD2VEC Google Sentiment      ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
    //      .similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleSentimentMapper))))
    //
    //    println("Balanced Synonymous with WORD2VEC Twitter Sentiment     ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
    //      .similarGeneratorAllWordsAllClasses(data, htf, word2vecFromTwitterSentimentMapper))))


    println("Balanced Scenario with BLANKOUT                         ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.balanced10foldWithBlankout(data,  htf , k, sentiwordSet))

    //    ---------------------------------------------- Imbalance & Over/under-Sampling -----------------------------------
    //
    //    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.unbalance10fold(dataSplit))
    //
    //    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.unbalance10foldWithOversampling(data, htf))
    //
    //    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    //    print_results(syntheticGenerationFunctions.unbalance10foldWithUndersampling(data, htf, sc))


    //    ------------------------------------------BLANKOUT Imbalance----------------------------------------------------
    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation-----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinority(data, htf, k, sentiwordSet))

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(data, htf, sc, k, sentiwordSet))

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(data, htf, sc, k, sentiwordSet))


    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation-----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClasses(data, htf, k, sentiwordSet))

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSampling(data, htf, sc, k, sentiwordSet))

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSampling(data, htf, sc, k, sentiwordSet))


    sys.exit(1)
    //    ------------------------------------------ANTONYMOUS Imbalance----------------------------------------------------
    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation         -----------------------------------")
      print_results(syntheticGenerationFunctions.unbalance10foldWithAntonyms(data, htf, wordnetMapper, i))

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation----------")
      print_results(syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSampling(data, htf, wordnetMapper, i, sc))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation-----------")
      print_results(syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSampling(data, htf, wordnetMapper, i, sc))
    }


    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data, htf, wordnetMapper, 0, sc))

    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data, htf, wordnetMapper, 2, sc))

    println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data, htf, wordnetMapper, 4, sc))

    //    -------------------------------------------------------------------------------------------------------------------
    //    -------------------------- WORD2VEC NO-SENTIMENT EMBEDDINGS ------------------------------------------------------------
    //    ----------------------------------------------------- GOOGLE NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES  --------------------------------------------------")
    print_results (syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data, htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data, htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data, htf, word2vecFromGoogleMapper))


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data, htf, word2vecFromGoogleMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromGoogleMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromGoogleMapper,sc))


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromGoogleMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data,  htf, word2vecFromGoogleMapper, sc))

    //----------------------------------------------------- TWITTER NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES  --------------------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, htf, word2vecFromTwitterMapper))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromTwitterMapper,sc))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromTwitterMapper,sc))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromTwitterMapper, sc))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromTwitterMapper, sc))

    //-------------------------------------------------------------------------------------------------------------------
    //------------------------ WORD2VEC SENTIMENT EMBEDDINGS  ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Google- Sentiment----------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Google- Sentiment------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS Google- Sentiment -------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data,htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Google- Sentiment --------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, htf, word2vecFromGoogleSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromGoogleSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data,htf, word2vecFromGoogleSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data,htf, word2vecFromGoogleSentimentMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Google- Sentiment --")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromGoogleSentimentMapper, sc))

    //----------------------------------------------------- TWITTER SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Twitter- Sentiment---------------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
      data, htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Twitter- Sentiment-----------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
      data, htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  Twitter- Sentiment------------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
      data, htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Twitter- Sentiment--------------------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
      data, htf, word2vecFromTwitterSentimentMapper))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment-----")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromTwitterSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  Twitter- Sentiment------")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromTwitterSentimentMapper,sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING Twitter- Sentiment--")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data, htf, word2vecFromTwitterSentimentMapper, sc))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING Twitter- Sentiment---")
    print_results(syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data, htf, word2vecFromTwitterSentimentMapper, sc))


  }

  def main(args: Array[String]) {

    def conf = new SparkConf().setAppName(this.getClass.getName)
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val sentiwordSet = new SentiwordLoader().getDictionary

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
    val options = args(1).toInt
    val blaknout_k_number = args(2).toInt
    // assign words to numbers converter
    val htf = new HashingTF(1500000)

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
          LabeledPoint(id, htf.transform(parts(2).toLowerCase().split(' ')))}.cache()

        distant_supervision_evaluation(syntheticGenerationFunctions, dataSet, data, htf, sc, blaknout_k_number, sentiwordSet)

      case 2 =>
        //-----------------------SemEval----------------------------------

        val finalsemEvalData = data.filter{line=>
          val parts = line.split(',')
          parts(1).equals("negative")
        }.union(sc.parallelize(data.filter{line=>
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


          LabeledPoint(id, htf.transform(parts(2).toLowerCase().split(' ')))
        }.cache()
        //0 for positive 1 for negative
        //        println("positive: " + dataSetSemEval.filter(_.label == 0.0).count)
        //        println("negative: " + dataSetSemEval.filter(_.label == 1.0).count)

        // force balance into the dataset by dropping positive instances
        //        val temp_balanced = dataSetSemEval.filter(_.label == 1.0).union(sc.parallelize(dataSetSemEval.filter(_.label == 0.0).take(11427)))

        // split dataset for hold out eval
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSetSemEval, finalsemEvalData, htf, sc, blaknout_k_number, sentiwordSet)

      case 3 =>
        //-----------------------TSentiment15----------------------------------

        val finalTSentiment15 = data.filter{line=>
          val parts = line.split(',')
          parts(1).equals("negative")
        }.union(sc.parallelize(data.filter{line=>
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
          LabeledPoint(id, htf.transform(parts(2).toLowerCase().split(' ')))}.cache()
        distant_supervision_evaluation(syntheticGenerationFunctions, dataSetTSentiment15, finalTSentiment15, htf, sc, blaknout_k_number, sentiwordSet)

    }

  }
}

