import java.util

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
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
    //    println("precision_negative: " + test_map("precision_negative"))
    //    println("precision_positive: " + test_map("precision_positive"))

    //    println("recall_positive: " + test_map("recall_positive"))
    //    println("recall_negative: " + test_map("recall_negative"))

    //    println(test_map("f1_positive"))
    //    println(test_map("f1_negative"))
    println(test_map("accuracy"))
    //    println(test_map("AucRoc"))
    //println("f1_positive: " + test_map("f1_positive"))
    //    println("f1_negative: " + test_map("f1_negative"))
    //
    //    println("accuracy: " + test_map("accuracy"))
    //    println("AucRoc: " + test_map("AucRoc"))

  }
/*

  def distant_supervision_evaluation_SVM(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: RDD[LabeledPoint],
                                         data_positive: RDD[String], data_negative: RDD[String], htf: HashingTF, sc: SparkContext,
                                         k: Int, sentiwordSet: util.HashSet[String], only_for_original_tests: RDD[String]) : Unit = {


    val original_evaluation_set = MLUtils.kFold(dataSplit, 10, System.currentTimeMillis().toInt)
    println("Positives : " + data_positive.count())
    println("Negatives : " + data_negative.count())
    //    //        ------------------------------------------ Balanced Scenarios ------------------------------------------------------------
    println("Simple Balanced Scenario                                ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit, original_evaluation_set))
    //
    for(i <- 0 to 5) {
      println("Balanced Scenario with Antonymous Scenario = " + i + "  -------------------------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit.union(syntheticGenerationFunctions
        .choose_antonymous_generation_function(only_for_original_tests, htf, wordnetMapper, i)), original_evaluation_set))
    }
    println("Balanced Synonymous with Google                         ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit.union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(
      only_for_original_tests,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)),
      original_evaluation_set)
    )
    println("Balanced Synonymous with Twitter               ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(
        only_for_original_tests,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet)),
      original_evaluation_set)
    )
    println("Balanced Synonymous with WORD2VEC Google Sentiment      ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(
        only_for_original_tests,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet)),
      original_evaluation_set)
    )
    println("Balanced Synonymous with WORD2VEC Twitter Sentiment     ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(
        only_for_original_tests,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet)),
      original_evaluation_set)
    )
    println("Balanced Scenario with BLANKOUT                         ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(syntheticGenerationFunctions.
      balanced10foldWithBlankout(
        only_for_original_tests,
        htf,
        k,
        sentiwordSet),
      original_evaluation_set)
    )
    //        ---------------------------------------------- Imbalance & Over/under-Sampling -----------------------------------
    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.imbalance10fold(
        data_positive,
        data_negative,
        htf),
      original_evaluation_set)
    )
    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithOversampling(
        data_positive,
        data_negative,
        htf,
        sc),
      original_evaluation_set)
    )
    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithUndersampling(
        data_positive,
        data_negative,
        htf,
        sc),
      original_evaluation_set)
    )

    //        ------------------------------------------BLANKOUT Imbalance----------------------------------------------------
    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation          -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinority(
        data_positive,
        data_negative,
        htf,
        k,
        sentiwordSet,
        original_evaluation_set),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        sc,
        k,
        sentiwordSet),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(
        data_positive,
        data_negative,
        htf,
        sc,
        k,
        sentiwordSet),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation  -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClasses(
        data_positive,
        data_negative,
        htf,
        k,
        sentiwordSet),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        sc,
        k,
        sentiwordSet),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation   -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSampling(
        data_positive,
        data_negative,
        htf,
        sc,
        k,
        sentiwordSet),
      original_evaluation_set)
    )

    //        ------------------------------------------BLANKOUT Imbalance NO FILTER FAIR EVALUATION----------------------------------------------------
    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation No filter - !!!-----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityNoFilter(
        data_positive,
        data_negative,
        htf,
        k,
        original_evaluation_set),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation No filter!!!  -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSamplingNoFilter(
        data_positive,
        data_negative,
        htf,
        sc,
        k),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation No filter!!!  -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSamplingNoFilter(
        data_positive,
        data_negative,
        htf,
        sc,
        k),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation No filter!!! -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesNoFilter(
        data_positive,
        data_negative,
        htf,
        k),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation No filter!!!  -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSamplingNoFilter(
        data_positive,
        data_negative,
        htf,
        sc,
        k),
      original_evaluation_set)
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation No filter!!!  -----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSamplingNoFilter(
        data_positive,
        data_negative,
        htf,
        sc,
        k),
      original_evaluation_set)
    )


    //        ------------------------------------------BLANKOUT Imbalance NO FILTER BIAS EVALUATION----------------------------------------------------
    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation No filter - !!!-----------------------------------")
    var blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityNoFilter(data_positive, data_negative, htf, k, original_evaluation_set)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt) )
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation No filter!!!  -----------------------------------")
    blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, k)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt) )
    )

    println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation No filter!!!  -----------------------------------")
    blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, k)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt) )
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation No filter!!! -----------------------------------")
    blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesNoFilter(data_positive, data_negative, htf, k)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt))
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation No filter!!!  -----------------------------------")
    blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, k)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt) )
    )

    println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation No filter!!!  -----------------------------------")
    blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, k)
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      blank_set ,
      MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt) )
    )


    //      //    ------------------------------------------ANTONYMOUS Imbalance----------------------------------------------------
    for(i <- 0 to 5) {
      println("Imbalanced Scenario with Antonymous No Balance After Generation         -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonyms(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          i),
        original_evaluation_set)
      )

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation----------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSampling(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          i,
          sc),
        original_evaluation_set)
      )

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation-----------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSampling(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          i,
          sc),
        original_evaluation_set)
      )

    }


    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation Filtered (using sentiwordnet as well)        -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsFiltered(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          sentiwordSet,
          i),
        original_evaluation_set)
      )

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)----------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSamplingFiltered(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          sentiwordSet,
          i,
          sc),
        original_evaluation_set)
      )

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)-----------")
      print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSamplingFiltered(
          data_positive,
          data_negative,
          htf,
          wordnetMapper,
          sentiwordSet,
          i,
          sc),
        original_evaluation_set)
      )
    }

    /*
        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 0, sc),original_evaluation_set))
        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 2, sc),original_evaluation_set))
        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 4, sc),original_evaluation_set))
    */

    //    -------------------------------------------------------------------------------------------------------------------
    //    -------------------------- WORD2VEC NO-SENTIMENT EMBEDDINGS ------------------------------------------------------------
    //    ----------------------------------------------------- GOOGLE NO-SENTIMENT EMBEDDINGS  ------------------------------------------
    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES  --------------------------------------------------")
    print_results (syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    //----------------------------------------------------- TWITTER NO-SENTIMENT EMBEDDINGS  ------------------------------------------
    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES  --------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    //-------------------------------------------------------------------------------------------------------------------
    //------------------------ WORD2VEC SENTIMENT EMBEDDINGS  ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE EMBEDDINGS  ------------------------------------------
    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Google- Sentiment----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Google- Sentiment------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS Google- Sentiment -------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Google- Sentiment --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Google- Sentiment --")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    //----------------------------------------------------- TWITTER SENTIMENT EMBEDDINGS  ------------------------------------------
    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Twitter- Sentiment---------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Twitter- Sentiment-----------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  Twitter- Sentiment------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Twitter- Sentiment--------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment-----")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  Twitter- Sentiment------")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING Twitter- Sentiment--")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )
    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING Twitter- Sentiment---")
    print_results(syntheticGenerationFunctions.cross_validation_function_second_classifier(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet),
      original_evaluation_set)
    )

  }*/

  def distant_supervision_evaluation_naive_bayes_fair_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: RDD[LabeledPoint],
                                                                 data_positive: RDD[String], data_negative: RDD[String], htf: HashingTF, sc: SparkContext,
                                                                 k: Int, sentiwordSet: util.HashSet[String], only_for_original_tests: RDD[String],
                                                                 percentage: Double) : Unit = {
    val original_evaluation_set = MLUtils.kFold(dataSplit, 10, System.currentTimeMillis().toInt)
    println("Positives : " + data_positive.count())
    println("Negatives : " + data_negative.count())

/*
    println("Simple Balanced Scenario                                ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit, original_evaluation_set))

    for(i <- 0 to 5) {
      println("Balanced Scenario with Antonymous Scenario = " + i + "  -------------------------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
        .choose_antonymous_generation_function(only_for_original_tests, htf, wordnetMapper, i)), original_evaluation_set))
    }

    println("Balanced Synonymous with Google                         ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(
      only_for_original_tests,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)),
      original_evaluation_set)
    )


    println("Balanced Synonymous with Twitter               ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet)),
      original_evaluation_set)
    )

    println("Balanced Synonymous with WORD2VEC Google Sentiment      ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet)),
      original_evaluation_set)
    )

    println("Balanced Synonymous with WORD2VEC Twitter Sentiment     ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromTwitterSentimentMapper, sentiwordSet)),
      original_evaluation_set)
    )

    println("Balanced Scenario with BLANKOUT                         ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(syntheticGenerationFunctions.
      balanced10foldWithBlankout(only_for_original_tests,htf , k, sentiwordSet),original_evaluation_set))

    //        ---------------------------------------------- Imbalance & Over/under-Sampling -----------------------------------
    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.imbalance10fold(data_positive, data_negative, htf),original_evaluation_set))

    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithOversampling(data_positive, data_negative, htf, sc),original_evaluation_set))

    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithUndersampling(data_positive, data_negative, htf, sc),original_evaluation_set))
*/

/*
    for (number<- 2 to 5) {
      println("THIS IS THE NUMBER FOR THIS ITERATION ~~~~~> " + number)
      //        ------------------------------------------BLANKOUT Imbalance----------------------------------------------------
      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation filtered -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinority(data_positive, data_negative, htf, number, sentiwordSet, original_evaluation_set), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation filtered   -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(data_positive, data_negative, htf, sc, number, sentiwordSet), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation filtered   -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(data_positive, data_negative, htf, sc, number, sentiwordSet), original_evaluation_set))


      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation filtered  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClasses(data_positive, data_negative, htf, number, sentiwordSet), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation filtered   -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSampling(data_positive, data_negative, htf, sc, number, sentiwordSet), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation  filtered  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSampling(data_positive, data_negative, htf, sc, number, sentiwordSet), original_evaluation_set))


      //        ------------------------------------------BLANKOUT Imbalance NO FILTER FAIR EVALUATION----------------------------------------------------
      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation No filter - !!!-----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityNoFilter(data_positive, data_negative, htf, number, original_evaluation_set), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation No filter!!!  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, number), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation No filter!!!  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, number), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation No filter!!! -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesNoFilter(data_positive, data_negative, htf, number), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation No filter!!!  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, number), original_evaluation_set))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation No filter!!!  -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, number), original_evaluation_set))

    }
*/

    //    ------------------------------------------ANTONYMOUS Imbalance----------------------------------------------------
   /* for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation         -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonyms(data_positive, data_negative, htf, wordnetMapper, i),original_evaluation_set))

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation----------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSampling(data_positive, data_negative, htf, wordnetMapper, i, sc),original_evaluation_set))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation-----------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSampling(data_positive, data_negative, htf, wordnetMapper, i, sc),original_evaluation_set))
    }

    //    ------------------------------------------ANTONYMOUS Imbalance Filtered (using sentiwordnet as well)---------------------------------

    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation Filtered (using sentiwordnet as well)        -----------------------------------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet, i),original_evaluation_set))

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)----------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSamplingFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet, i, sc),original_evaluation_set))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)-----------")
      print_results(syntheticGenerationFunctions.cross_validation_function(
        syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSamplingFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet,i, sc),original_evaluation_set))
    }
*/

    //    -------------------------------------------------------------------------------------------------------------------
    //    -------------------------- WORD2VEC NO-SENTIMENT EMBEDDINGS ------------------------------------------------------------
    //    ----------------------------------------------------- GOOGLE NO-SENTIMENT EMBEDDINGS  ------------------------------------------

   println("Imbalanced Synonymous with Google ALL Words ALL CLASSES  --------------------------------------------------")
    print_results (syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    //----------------------------------------------------- TWITTER NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES  --------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES -----------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    //-------------------------------------------------------------------------------------------------------------------
    //------------------------ WORD2VEC SENTIMENT EMBEDDINGS  ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE EMBEDDINGS  ------------------------------------------
    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Google- Sentiment----------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Google- Sentiment------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS Google- Sentiment -------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Google- Sentiment --------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Google- Sentiment -----")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Google- Sentiment --")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromGoogleSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


    //----------------------------------------------------- TWITTER SENTIMENT EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Twitter- Sentiment---------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Twitter- Sentiment-----------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  Twitter- Sentiment------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Twitter- Sentiment--------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sentiwordSet,
        sc,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment-----")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  Twitter- Sentiment------")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING Twitter- Sentiment--")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING Twitter- Sentiment---")
    print_results(syntheticGenerationFunctions.cross_validation_function(
      syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
        data_positive,
        data_negative,
        htf,
        word2vecFromTwitterSentimentMapper,
        sc,
        sentiwordSet,
        percentage),
      original_evaluation_set)
    )


  }



  //---------------------------------------------------------------------
  def distant_supervision_evaluation_naive_bayes_bias_evaluation(syntheticGenerationFunctions: SyntheticGenerationFunctions, dataSplit: RDD[LabeledPoint],
                                                 data_positive: RDD[String], data_negative: RDD[String], htf: HashingTF, sc: SparkContext,
                                                 k: Int, sentiwordSet: util.HashSet[String], only_for_original_tests: RDD[String]) : Unit = {

    val original_evaluation_set = MLUtils.kFold(dataSplit, 10, System.currentTimeMillis().toInt)
    println("Positives : " + data_positive.count())
    println("Negatives : " + data_negative.count())

/*

    //        ------------------------------------------ Balanced Scenarios ------------------------------------------------------------
    println("Simple Balanced Scenario                                ---------------------------------------------------")
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit, original_evaluation_set))
    //
    for(i <- 0 to 5) {
      println("Balanced Scenario with Antonymous Scenario = " + i + "  -------------------------------------------------")
      var temp = dataSplit.union(syntheticGenerationFunctions
        .choose_antonymous_generation_function(only_for_original_tests, htf, wordnetMapper, i))
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt)))
    }

    println("Balanced Synonymous with Google                         ---------------------------------------------------")
    var temp = syntheticGenerationFunctions.similarGeneratorAllWordsAllClasses(
      only_for_original_tests,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(temp),
      MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt))
    )


    println("Balanced Synonymous with Twitter               ---------------------------------------------------")
    temp = syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromTwitterMapper,
        sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(temp),
      MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt))
    )

    println("Balanced Synonymous with WORD2VEC Google Sentiment      ---------------------------------------------------")
    temp = syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromGoogleSentimentMapper,
        sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(temp),
      MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt))
    )

    println("Balanced Synonymous with WORD2VEC Twitter Sentiment     ---------------------------------------------------")
    temp = syntheticGenerationFunctions
      .similarGeneratorAllWordsAllClasses(only_for_original_tests,
        htf,
        word2vecFromTwitterSentimentMapper, sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(dataSplit.union(temp),
      MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt))
    )

    println("Balanced Scenario with BLANKOUT                         ---------------------------------------------------")
    temp = syntheticGenerationFunctions.balanced10foldWithBlankout(only_for_original_tests, htf, k, sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt)))

    //        ---------------------------------------------- Imbalance & Over/under-Sampling -----------------------------------
    println("Simple Imbalanced Scenario            ---------------------------------------------------")
    temp = syntheticGenerationFunctions.imbalance10fold(data_positive, data_negative, htf)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp , MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt)))

    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    temp =syntheticGenerationFunctions.unbalance10foldWithOversampling(data_positive, data_negative, htf, sc)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp , MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt)))

    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithUndersampling(data_positive, data_negative, htf, sc)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp , MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt)))


*/
    for (number<- 2 to 5) {
      println("THIS IS THE NUMBER FOR THIS ITERATION ~~~~~> " + number )

      //        ------------------------------------------BLANKOUT Imbalance FILTER ----------------------------------------------------
      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation filtered - !!!-----------------------------------")
      var blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinority(data_positive, data_negative, htf, number, sentiwordSet, original_evaluation_set)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation filtered!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(data_positive, data_negative, htf, sc, number, sentiwordSet)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation filtered!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(data_positive, data_negative, htf, sc, number, sentiwordSet)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation filtered!!! -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClasses(data_positive, data_negative, htf, number, sentiwordSet)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation filtered!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSampling(data_positive, data_negative, htf, sc, number, sentiwordSet)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation filtered!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSampling(data_positive, data_negative, htf, sc, number, sentiwordSet)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))


      //        ------------------------------------------BLANKOUT Imbalance NO FILTER ----------------------------------------------------
      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY No Balance After Generation No filter - !!!-----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityNoFilter(data_positive, data_negative, htf, number, original_evaluation_set)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Undersampling After Generation No filter!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, number)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT ONLY MINORITY with Oversampling After Generation No filter!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutOnlyMinorityWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, number)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES No Balance After Generation No filter!!! -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesNoFilter(data_positive, data_negative, htf, number)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Undersampling After Generation No filter!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithUnderSamplingNoFilter(data_positive, data_negative, htf, sc, number)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))

      println("Imbalanced Scenario with BLANKOUT BOTH CLASSES with Oversampling After Generation No filter!!!  -----------------------------------")
      blank_set = syntheticGenerationFunctions.unbalance10foldWithBlankoutBothClassesWithOverSamplingNoFilter(data_positive, data_negative, htf, sc, number)
      print_results(syntheticGenerationFunctions.cross_validation_function(blank_set, MLUtils.kFold(blank_set, 10, System.currentTimeMillis().toInt)))
    }
/*

    //    ------------------------------------------ANTONYMOUS Imbalance----------------------------------------------------
    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation         -----------------------------------")
      var temp = syntheticGenerationFunctions.unbalance10foldWithAntonyms(data_positive, data_negative, htf, wordnetMapper, i)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation----------")
      temp = syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSampling(data_positive, data_negative, htf, wordnetMapper, i, sc)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation-----------")
      temp = syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSampling(data_positive, data_negative, htf, wordnetMapper, i, sc)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))
    }

    //    ------------------------------------------ANTONYMOUS Imbalance Filtered (using sentiwordnet as well)----------------

    for(i <- 0 to 5) {

      println("Imbalanced Scenario with Antonymous No Balance After Generation Filtered (using sentiwordnet as well)        -----------------------------------")
      var temp = syntheticGenerationFunctions.unbalance10foldWithAntonymsFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet, i)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


      println("Imbalanced Scenario with Antonymous with Re-Balance -UNDERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)----------")
      temp = syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceUnderSamplingFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet, i, sc)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

      println("Imbalanced Scenario with Antonymous with Re-Balance -OVERSamplingMajorityClass- After Generation Filtered (using sentiwordnet as well)-----------")
      temp = syntheticGenerationFunctions.unbalance10foldWithAntonymsAndRebalanceOverSamplingFiltered(data_positive, data_negative, htf, wordnetMapper, sentiwordSet,i, sc)
      print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    }

*/

    /*
        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 0, sc),original_evaluation_set))

        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 2, sc),original_evaluation_set))

        println("Imbalanced Scenario with Antonymous with Perturbation of Majority Class    --------------------------------")
        print_results(syntheticGenerationFunctions.cross_validation_function(
          syntheticGenerationFunctions.unbalance10foldWithAntonymsFairPerturbation(data_positive, data_negative, htf, wordnetMapper, 4, sc),original_evaluation_set))
    */


    //    -------------------------------------------------------------------------------------------------------------------
    //    -------------------------- WORD2VEC NO-SENTIMENT EMBEDDINGS ------------------------------------------------------------
    //    ----------------------------------------------------- GOOGLE NO-SENTIMENT EMBEDDINGS  ------------------------------------------

    /*

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES  --------------------------------------------------")
    var temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES -----------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
        data_negative,
        htf,
        word2vecFromGoogleMapper,
        sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    //----------------------------------------------------- TWITTER NO-SENTIMENT EMBEDDINGS  ------------------------------------------


    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES  --------------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES -----------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS  ------------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  --------------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  -----------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING  ------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING --------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING ---------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    //-------------------------------------------------------------------------------------------------------------------
    //------------------------ WORD2VEC SENTIMENT EMBEDDINGS  ------------------------------------------------------------
    //----------------------------------------------------- GOOGLE EMBEDDINGS  ------------------------------------------

    println("Imbalanced Synonymous with Google ALL Words ALL CLASSES Google- Sentiment----------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Google ONLY ONE Word ALL CLASSES Google- Sentiment------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS Google- Sentiment -------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS  Google- Sentiment --------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -----")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Google- Sentiment -----")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Google- Sentiment -")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Google ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Google- Sentiment --")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromGoogleSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    //----------------------------------------------------- TWITTER SENTIMENT EMBEDDINGS  ------------------------------------------


    println("Imbalanced Synonymous with Twitter ALL Words ALL CLASSES Twitter- Sentiment----------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ALL CLASSES Twitter- Sentiment------------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordAllClasses(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS Twitter- Sentiment -------------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS  Twitter- Sentiment --------------------")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment -----")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ALL Words ONLY MINORITY CLASS WITH OVERSAMPLING   Twitter- Sentiment -----")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH UNDERSAMPLING  Twitter- Sentiment -")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))

    println("Imbalanced Synonymous with Twitter ONLY ONE Word ONLY MINORITY CLASS WITH OVERSAMPLING  Twitter- Sentiment --")
    temp = syntheticGenerationFunctions.unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(
      data_positive,
      data_negative,
      htf,
      word2vecFromTwitterSentimentMapper,
      sc,
      sentiwordSet)
    print_results(syntheticGenerationFunctions.cross_validation_function(temp, MLUtils.kFold(temp, 10, System.currentTimeMillis().toInt) ))


*/

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
    val percentage = args(3).toDouble
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
          LabeledPoint(id, htf.transform(parts(2).split(' ')))}.cache()

        val positive = data.filter(x => x.split(",")(1).equals("positive"))
        val negative = data.filter(x => x.split(",")(1).equals("negative")).randomSplit(Array(0.8, 0.2))(1)

        println("\n\n\n ---------------FAIR Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_fair_evaluation(syntheticGenerationFunctions, dataSet,
          positive,
          negative,
          htf, sc, blaknout_k_number, sentiwordSet, data,percentage)

/*        println("\n\n\n ---------------BIASED Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_bias_evaluation(syntheticGenerationFunctions, dataSet,
          positive,
          negative,
          htf, sc, blaknout_k_number, sentiwordSet, data)*/

      case 2 =>
        // -----------------------SemEval----------------------------------

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


          LabeledPoint(id, htf.transform(parts(2).split(' ')))
        }.cache()
        val positive = finalsemEvalData.filter(x => x.split(",")(1).equals("positive"))
        val negative = finalsemEvalData.filter(x => x.split(",")(1).equals("negative")).randomSplit(Array(0.8, 0.2))(1)

        println("\n\n\n ---------------FAIR Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_fair_evaluation(syntheticGenerationFunctions, dataSetSemEval,
          positive,
          negative,
          htf, sc, blaknout_k_number, sentiwordSet, finalsemEvalData,percentage)

/*
        println("\n\n\n ---------------BIASED Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_bias_evaluation(syntheticGenerationFunctions, dataSetSemEval,
          positive,
          negative,
          htf, sc, blaknout_k_number, sentiwordSet, finalsemEvalData)
*/

      case 3 =>
        // -----------------------TSentiment15----------------------------------

        val dataSet = data.map { line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(1).equals("positive")) {
            id = 0.0
          }
          LabeledPoint(id, htf.transform(parts(2).split(' ')))}.cache()

        println("\n\n\n ---------------FAIR Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_fair_evaluation(syntheticGenerationFunctions, dataSet,
          data.filter(x => x.split(",")(1).equals("positive")),
          data.filter(x => x.split(",")(1).equals("negative")),
          htf, sc, blaknout_k_number, sentiwordSet, data,percentage)

/*        println("\n\n\n ---------------BIASED Evaluation !!! -------------------------- \n\n\n")
        distant_supervision_evaluation_naive_bayes_bias_evaluation(syntheticGenerationFunctions, dataSet,
          data.filter(x => x.split(",")(1).equals("positive")),
          data.filter(x => x.split(",")(1).equals("negative")),
          htf, sc, blaknout_k_number, sentiwordSet, data)*/

    }

  }
}
