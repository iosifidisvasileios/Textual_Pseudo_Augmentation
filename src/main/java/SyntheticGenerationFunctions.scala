import java.util

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class SyntheticGenerationFunctions {

  def cross_validation_function(data:RDD[LabeledPoint], original_test_evaluation:Array[(RDD[LabeledPoint],RDD[LabeledPoint])]): mutable.Map [String, Double] = {

    val mapper = mutable.Map[String, Double]().withDefaultValue(0)

    println(data.filter(_.label == 0).count)
    println(data.filter(_.label == 1).count)

    val output_array = ArrayBuffer[ RDD[ (Double,Double) ]] ()
    var counter = 0

    val foldedDataSet = MLUtils.kFold(data, 10, System.currentTimeMillis().toInt)
    foldedDataSet.foreach{item =>

      val model = NaiveBayes.train(item._1)
      val predictionAndLabels = original_test_evaluation(counter)._2.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
      output_array.append(predictionAndLabels)
      counter += 1
    }

    var positive_counter = 0.0
    var negative_counter = 0.0

    output_array.foreach{ x=>
      val metrics = new BinaryClassificationMetrics(x)
      val f1Score = metrics.fMeasureByThreshold.collectAsMap()
      if (f1Score.contains(0.0)){
        positive_counter += 1
        mapper("f1_positive") += f1Score.get(0.0).last
      }
      if (f1Score.contains(1.0)){
        negative_counter += 1
        mapper("f1_negative") += f1Score.get(1.0).last
      }
      val auPRC = metrics.areaUnderPR
      mapper("accuracy") += (auPRC / 10)
      val auROC = metrics.areaUnderROC
      mapper("AucRoc") += (auROC / 10)
    }
    mapper("f1_positive") = mapper("f1_positive") / positive_counter
    mapper("f1_negative") = mapper("f1_negative") / negative_counter

    mapper
  }

  def cross_validation_function_second_classifier(data:RDD[LabeledPoint],
                                                  original_test_evaluation:Array[(RDD[LabeledPoint], RDD[LabeledPoint])]): mutable.Map [String, Double] = {

    val mapper = mutable.Map[String, Double]().withDefaultValue(0)

    println(data.filter(_.label == 0).count)
    println(data.filter(_.label == 1).count)

    val output_array = ArrayBuffer[ RDD[ (Double,Double) ]] ()
    var counter = 0

    val foldedDataSet = MLUtils.kFold(data, 10, System.currentTimeMillis().toInt)
    foldedDataSet.foreach{item =>

      val model =  SVMWithSGD.train(item._1 , 1)
      val predictionAndLabels = original_test_evaluation(counter)._2.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
      output_array.append(predictionAndLabels)
      counter += 1
    }

    var positive_counter = 0.0
    var negative_counter = 0.0

    output_array.foreach{ x=>
      val metrics = new BinaryClassificationMetrics(x)
      val f1Score = metrics.fMeasureByThreshold.collectAsMap()
      if (f1Score.contains(0.0)){
        positive_counter += 1
        mapper("f1_positive") += f1Score.get(0.0).last
      }
      if (f1Score.contains(1.0)){
        negative_counter += 1
        mapper("f1_negative") += f1Score.get(1.0).last
      }
      val auPRC = metrics.areaUnderPR
      mapper("accuracy") += (auPRC / 10)
      val auROC = metrics.areaUnderROC
      mapper("AucRoc") += (auROC / 10)
    }
    mapper("f1_positive") = mapper("f1_positive") / positive_counter
    mapper("f1_negative") = mapper("f1_negative") / negative_counter

    mapper
  }

  def simple10fold(data: RDD[LabeledPoint], test: RDD[LabeledPoint]): RDD[(Double,Double)] = {
    //    cross_validation_function(data)
    //    println("Training set: positives before antonym generation = " + data.filter(_.label == 0).count())
    //    println("Training set: negatives before antonym generation = " + data.filter(_.label == 1).count())

    //    println("Testing set: positives before antonym generation = "  + test.filter(_.label == 0).count())
    //    println("Testing set: negatives before antonym generation = "  + test.filter(_.label == 1).count())

    val model = NaiveBayes.train(data)
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def imbalance10fold(data_pos: RDD[String], data_neg: RDD[String], htf:HashingTF): RDD[LabeledPoint] = {

    data_neg.union(data_pos).map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

  }

  def similarGeneratorOnlyOneWordAllClasses(data: RDD[String],
                                            htf: HashingTF,
                                            mapper : mutable.HashMap[String, String],
                                            sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    val synonymous = data.map { line =>
      val line2 = line.toLowerCase
      val parts = line2.split(",")
      val text = line2.split(",")(2)
      var unigramsSynonymous = ""
      var one_word_flag = false
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
        if(!one_word_flag){
          if (mapper.contains(word)){
//          if (mapper.contains(word) && sentiwordSet.contains(word.toLowerCase)){
            similarityFlag = true
            one_word_flag = true
            //            println("it is true! " + mapper(word))
            unigramsSynonymous  += mapper(word) + " "
          }else{
            unigramsSynonymous += word + " "
          }
        }else{
          unigramsSynonymous += word + " "
        }
      }

      if(similarityFlag ){
        status + "," + unigramsSynonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("instances = " + synonymous.count())
    //    println("positives generated = " + synonymous.filter(_.label == 0).count())
    //    println("negatives generated = " + synonymous.filter(_.label == 1).count())
    synonymous
  }

  def similarGeneratorOnlyOneWordOnlyMinorityClass(data: RDD[String],
                                                   htf: HashingTF,
                                                   mapper : mutable.HashMap[String, String],
                                                   sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    val synonymous = data.map { line =>
      val line2 = line.toLowerCase
      val parts = line2.split(",")
      val text = line2.split(",")(2)
      var unigramsSynonymous = ""
      var one_word_flag = false
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
        if(!one_word_flag){
          if (mapper.contains(word) ){
//          if (mapper.contains(word) && sentiwordSet.contains(word.toLowerCase)){
            similarityFlag = true
            one_word_flag = true
            unigramsSynonymous  += mapper(word) + " "
          }else{
            unigramsSynonymous += word + " "
          }
        }else{
          unigramsSynonymous += word + " "
        }
      }

      if(similarityFlag ){
        status + "," + unigramsSynonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }

    synonymous.filter(_.label == 1)
  }

  def similarGeneratorAllWordsAllClasses(data: RDD[String],
                                         htf: HashingTF,
                                         mapper : mutable.HashMap[String, String],
                                         sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    //    println("size of dictionary : " + mapper.size)

    val synonymous = data.map { line =>
      val line2 = line.toLowerCase
      val parts = line2.split(",")
      val text = line2.split(",")(2)
      var unigramsSynonymous = ""
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
        if (mapper.contains(word) ){
//        if (mapper.contains(word) && sentiwordSet.contains(word.toLowerCase)){
          similarityFlag = true
          //          println("it is true! " + mapper(word))
          unigramsSynonymous  += mapper(word) + " "
        }else{
          unigramsSynonymous += word + " "
        }
      }

      if(similarityFlag ){
        status + "," + unigramsSynonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("instances = " + synonymous.count())
    //    println("positives generated = " + synonymous.filter(_.label == 0).count())
    //    println("negatives generated = " + synonymous.filter(_.label == 1).count())
    //    denoise_synthetic_data(synonymous)
    synonymous
  }

  def denoise_synthetic_data_with_init_set(data: RDD[String], synthetic_data: RDD[LabeledPoint] , htf: HashingTF): RDD[LabeledPoint] = {
    val finalDataset = synthetic_data.union(data.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val mnb_model = NaiveBayes.train(finalDataset)
    synthetic_data.filter( x=> mnb_model.predict(x.features) == x.label)
  }

  def denoise_synthetic_data(synthetic_data: RDD[LabeledPoint] ): RDD[LabeledPoint] = {
    val mnb_model = NaiveBayes.train(synthetic_data)
    synthetic_data.filter( x=> mnb_model.predict(x.features) == x.label)
  }

  def similarGeneratorAllWordsOnlyMinorityClass(data: RDD[String],
                                                htf: HashingTF,
                                                mapper : mutable.HashMap[String, String],
                                                sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {

    val synonymous = data.map { line =>
      val line2 = line.toLowerCase
      val parts = line2.split(",")
      val text = line2.split(",")(2)
      var unigramsSynonymous = ""
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
//        if (mapper.contains(word) && sentiwordSet.contains(word.toLowerCase)){
        if (mapper.contains(word) ){
          similarityFlag = true
          //          println("it is true! " + mapper(word))
          unigramsSynonymous  += mapper(word) + " "
        }else{
          unigramsSynonymous += word + " "
        }
      }

      if(similarityFlag ){
        status + "," + unigramsSynonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("instances = " + synonymous.count())
    //    println("positives generated = " + synonymous.filter(_.label == 0).count())
    //    println("negatives generated = " + synonymous.filter(_.label == 1).count())
    synonymous.filter(_.label == 1)
    //    val output = denoise_synthetic_data(data, synonymous, htf)
    //    println("positives after denoise = " + output.filter(_.label == 0).count())
    //    println("negatives after denoise = " + output.filter(_.label == 1).count())
    //    output
  }

  def similarGeneratorSemEval(data: RDD[String], htf: HashingTF, mapper : mutable.HashMap[String, String]): RDD[LabeledPoint] = {
    //    println("size of dictionary : " + mapper.size)

    val synonymous = data.map { line =>
      //      val line2 = line.toLowerCase
      val parts = line.split("\t")
      val text = parts(3).toLowerCase
      var unigramsSynonymous = ""
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
        if (mapper.contains(word)){
          similarityFlag = true
          //          println("it is true! " + mapper(word))
          unigramsSynonymous  += mapper(word) + " "
        }else{
          unigramsSynonymous += word + " "
        }
      }

      if(similarityFlag ){
        status + "," + unigramsSynonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("instances = " + synonymous.count())
    synonymous
  }


  def choose_antonymous_generation_function(unbalanced: RDD[String], htf: HashingTF, wordnetMapper: mutable.HashMap[String, String], antonymous_case: Int): RDD[LabeledPoint] = {

    val filtered_data = filter_antonyms_generation(unbalanced, wordnetMapper)

    antonymous_case match {
      //      case 0 =>
      //        println("case 0 for antonymous generation")
      //        antonymsGeneratorPaperBasedBothClasses(unbalanced, htf, wordnetMapper)
      case 0 =>
        println("case 0 for antonymous generation -- FilteredBothClasses")
        antonymsGeneratorPaperBasedBothClasses(filtered_data , htf, wordnetMapper)
      case 1 =>
        println("case 1 for antonymous generation -- FilteredOnlyImbalancedClass")
        antonymsGeneratorPaperBasedOnlyImbalancedClass(filtered_data , htf, wordnetMapper)

      case 2 =>
        println("case 2 for antonymous generation -- UnfilteredNegationsAndWordsBothClasses")
        antonymsGeneratorNegationsAndWordsBothClasses(unbalanced, htf, wordnetMapper)
      case 3 =>
        println("case 3 for antonymous generation -- UnfilteredNegationsAndWordsOnlyImbalancedClass")
        antonymsGeneratorNegationsAndWordsOnlyImbalancedClass(unbalanced, htf, wordnetMapper)

      case 4 =>
        println("case 4 for antonymous generation -- UnfilteredOnlyOneWordBothClasses")
        antonymsGeneratorOnlyOneWordBothClasses(unbalanced , htf, wordnetMapper)
      case 5 =>
        println("case 5 for antonymous generation -- UnfilteredOnlyOneWordOnlyImbalancedClass")
        antonymsGeneratorOnlyOneWordOnlyImbalancedClass(unbalanced , htf, wordnetMapper)
      //      case _ =>
      //        println("case 0 for antonymous generation")
      //        antonymsGeneratorPaperBasedBothClasses(filtered_data , htf, wordnetMapper)
    }

  }

  def choose_antonymous_generation_filtered_function(unbalanced: RDD[String],
                                                     htf: HashingTF,
                                                     wordnetMapper: mutable.HashMap[String, String],
                                                     sentiwordSet: util.HashSet[String],
                                                     antonymous_case: Int): RDD[LabeledPoint] = {

    val filtered_data = filter_antonyms_generation(unbalanced, wordnetMapper)

    antonymous_case match {
      //      case 0 =>
      //        println("case 0 for antonymous generation")
      //        antonymsGeneratorPaperBasedBothClasses(unbalanced, htf, wordnetMapper)
      case 0 =>
        println("case 0 for antonymous generation -- FilteredBothClasses")
        antonymsGeneratorFilteredPaperBasedBothClasses(filtered_data , htf, wordnetMapper, sentiwordSet)
      case 1 =>
        println("case 1 for antonymous generation -- FilteredOnlyImbalancedClass")
        antonymsGeneratorFilteredPaperBasedOnlyImbalancedClass(filtered_data , htf, wordnetMapper, sentiwordSet)

      case 2 =>
        println("case 2 for antonymous generation -- UnfilteredNegationsAndWordsBothClasses")
        antonymsGeneratorFilteredNegationsAndWordsBothClasses(unbalanced, htf, wordnetMapper, sentiwordSet)
      case 3 =>
        println("case 3 for antonymous generation -- UnfilteredNegationsAndWordsOnlyImbalancedClass")
        antonymsGeneratorFilteredNegationsAndWordsOnlyImbalancedClass(unbalanced, htf, wordnetMapper, sentiwordSet)

      case 4 =>
        println("case 4 for antonymous generation -- UnfilteredOnlyOneWordBothClasses")
        antonymsGeneratorFilteredOnlyOneWordBothClasses(unbalanced , htf, wordnetMapper, sentiwordSet)
      case 5 =>
        println("case 5 for antonymous generation -- UnfilteredOnlyOneWordOnlyImbalancedClass")
        antonymsGeneratorFilteredOnlyOneWordOnlyImbalancedClass(unbalanced , htf, wordnetMapper, sentiwordSet)
      //      case _ =>
      //        println("case 0 for antonymous generation")
      //        antonymsGeneratorPaperBasedBothClasses(filtered_data , htf, wordnetMapper)
    }

  }


  def blankout_generator(negatives: RDD[String], k: Int, sentiwordSet: util.HashSet[String]): RDD[String] = {
    val logger = LoggerFactory.getLogger("TheLoggerName")
    //    val r = new java.util.Random

    val filtered = negatives.filter(x => x.split(",")(2).split(" ").length - k >= 3)

    filtered.map{ line =>

      val parts = line.split(',')
      val number_of_words = parts(2).split(' ').length
      var tempString = ""
      val indexSet = scala.collection.mutable.Set[Int]()

      val sentence = parts(2).split(' ')

      var special_counter = 0

      while(special_counter <= 100 && indexSet.size != k ){
        val r = new java.util.Random

        special_counter += 1
        var random_dropout = r.nextInt(number_of_words)


        if (!sentence(random_dropout).startsWith("NOT_") &&
          !sentence(random_dropout).startsWith("not_") &&
          !sentence(random_dropout).startsWith("NOT") &&
          !sentence(random_dropout).equals("dont") &&
          !sentence(random_dropout).equals("havent") &&
          !sentence(random_dropout).equals("cant") &&
          !sentence(random_dropout).equals("cannot") &&
          !sentence(random_dropout).equals("arent") &&
          !sentence(random_dropout).equals("aint") &&
          !sentence(random_dropout).startsWith("not") &&
          !sentence(random_dropout).endsWith("not") &&
          !sentiwordSet.contains(sentence(random_dropout))){
          indexSet.add(random_dropout)

        }
      }

      var outputStr = ""
      if(indexSet.nonEmpty ) {
        var counter = 0
        for (word <- parts(2).split(" ")) {
          if (!indexSet.contains(counter)) {
            tempString += word + " "
          }
          counter += 1
        }
        outputStr = parts(0) + "," + parts(1) + "," + tempString.trim()
      }else{
        outputStr = "null"
      }
      outputStr
    }.filter(x=> !x.equals("null"))

  }

  def blankout_generator_no_filter(negatives: RDD[String],
                                   k: Int): RDD[String] = {
    val logger = LoggerFactory.getLogger("TheLoggerName")
    //    val r = new java.util.Random

    val filtered = negatives.filter(x => x.split(",")(2).split(" ").length - k >= 3)

    filtered.map{ line =>

      val parts = line.split(',')
      val number_of_words = parts(2).split(' ').length
      var tempString = ""
      val indexSet = scala.collection.mutable.Set[Int]()

      val sentence = parts(2).split(' ')

      var special_counter = 0

      while(special_counter <= 100 && indexSet.size != k ){
        val r = new java.util.Random

        special_counter += 1
        var random_dropout = r.nextInt(number_of_words)


        if (!sentence(random_dropout).startsWith("NOT_") &&
          !sentence(random_dropout).startsWith("not_") &&
          !sentence(random_dropout).startsWith("NOT") &&
          !sentence(random_dropout).equals("dont") &&
          !sentence(random_dropout).equals("havent") &&
          !sentence(random_dropout).equals("cant") &&
          !sentence(random_dropout).equals("cannot") &&
          !sentence(random_dropout).equals("arent") &&
          !sentence(random_dropout).equals("aint") &&
          !sentence(random_dropout).startsWith("not") &&
          !sentence(random_dropout).endsWith("not")){
          indexSet.add(random_dropout)

        }
      }

      var outputStr = ""
      if(indexSet.nonEmpty ) {
        var counter = 0
        for (word <- parts(2).split(" ")) {
          if (!indexSet.contains(counter)) {
            tempString += word + " "
          }
          counter += 1
        }
        outputStr = parts(0) + "," + parts(1) + "," + tempString.trim()
      }else{
        outputStr = "null"
      }
      outputStr
    }.filter(x=> !x.equals("null"))

  }

  def unbalance10foldWithBlankoutOnlyMinority(positive: RDD[String],
                                              negative: RDD[String],
                                              htf : HashingTF,
                                              k: Int,
                                              sentiwordSet:util.HashSet[String],
                                              original_test_evaluation:Array[(RDD[LabeledPoint],RDD[LabeledPoint])]): RDD[LabeledPoint]= {
    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    var positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val positiveTemp = positive.union(blankout_generator(negative, k, sentiwordSet )).union(negative)

    positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }
    //    println("Training set: positives before antonym generation = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives before antonym generation = " + finalDataset.filter(_.label == 1).count())
  }

  def unbalance10foldWithBlankoutOnlyMinorityNoFilter(positive: RDD[String],
                                                      negative: RDD[String],
                                                      htf : HashingTF,
                                                      k: Int,
                                                      original_test_evaluation:Array[(RDD[LabeledPoint],RDD[LabeledPoint])]): RDD[LabeledPoint]= {

    val positiveTemp = positive.union(blankout_generator_no_filter(negative, k)).union(negative)

    positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }
  }

  def unbalance10foldWithBlankoutBothClasses(positive: RDD[String],
                                             negative: RDD[String],
                                             htf : HashingTF,
                                             k: Int,
                                             sentiwordSet: util.HashSet[String]): RDD[LabeledPoint]= {
    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    var positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    //    println("positives before blankout = " + positiveTemp.count())
    //    println("negatives before blankout = " + negative(1).count())

    val blankGeneration = positive.union(blankout_generator(positive, k, sentiwordSet))
      .union(blankout_generator(negative, k, sentiwordSet))
      .union(negative)

    //    println("positives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("positive")).count())
    //    println("negatives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("negative")).count())

    blankGeneration.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }
    //    cross_validation_function(finalDataset, original_test_evaluation)
  }


  def unbalance10foldWithBlankoutBothClassesNoFilter(positive: RDD[String],
                                                     negative: RDD[String],
                                                     htf : HashingTF,
                                                     k: Int): RDD[LabeledPoint]= {

    val blankGeneration = positive.union(blankout_generator_no_filter(positive, k))
      .union(blankout_generator_no_filter(negative, k))
      .union(negative)

    blankGeneration.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }
  }


  def unbalance10foldWithBlankoutOnlyMinorityWithUnderSampling(positive: RDD[String],
                                                               negative: RDD[String],
                                                               htf : HashingTF,
                                                               sc:SparkContext,
                                                               k: Int,
                                                               sentiwordSet: util.HashSet[String]): RDD[LabeledPoint]= {

    val positiveTemp = positive.union(blankout_generator(negative, k , sentiwordSet)).union(negative)

    val tempDataset = positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val negative_rdd = tempDataset .filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    negative_rdd.union(sc.parallelize(tempDataset .filter( _.label == 0).take(negative_count.toInt)))
  }

  def unbalance10foldWithBlankoutOnlyMinorityWithUnderSamplingNoFilter(positive: RDD[String],
                                                                       negative: RDD[String],
                                                                       htf : HashingTF,
                                                                       sc:SparkContext,
                                                                       k: Int): RDD[LabeledPoint]= {

    val positiveTemp = positive.union(blankout_generator_no_filter(negative, k )).union(negative)

    val tempDataset = positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val negative_rdd = tempDataset .filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    negative_rdd.union(sc.parallelize(tempDataset .filter( _.label == 0).take(negative_count.toInt)))
  }

  def unbalance10foldWithBlankoutBothClassesWithUnderSampling(positive: RDD[String],
                                                              negative: RDD[String],
                                                              htf : HashingTF,
                                                              sc:SparkContext,
                                                              k: Int,
                                                              sentiwordSet: util.HashSet[String]): RDD[LabeledPoint]= {

    //    var negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    var positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val blankGeneration = positive.union(blankout_generator(positive, k, sentiwordSet))
      .union(blankout_generator(negative, k, sentiwordSet ))
      .union(negative)

    //    println("positives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("positive")).count())
    //    println("negatives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("negative")).count())


    val tempDataset = blankGeneration.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val negative_rdd = tempDataset.filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    //    println("total instances for pos/neg after UnderSampling = " + negative_count)
    negative_rdd.union(sc.parallelize(tempDataset .filter( _.label == 0).take(negative_count.toInt)))

    //    cross_validation_function(finalDataset, original_test_evaluation )
  }


  def unbalance10foldWithBlankoutBothClassesWithUnderSamplingNoFilter(positive: RDD[String],
                                                                      negative: RDD[String],
                                                                      htf : HashingTF,
                                                                      sc:SparkContext,
                                                                      k: Int): RDD[LabeledPoint]= {

    val blankGeneration = positive.union(blankout_generator_no_filter(positive, k))
      .union(blankout_generator_no_filter(negative, k))
      .union(negative)

    val tempDataset = blankGeneration.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val negative_rdd = tempDataset.filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    negative_rdd.union(sc.parallelize(tempDataset .filter( _.label == 0).take(negative_count.toInt)))
  }


  def unbalance10foldWithBlankoutOnlyMinorityWithOverSampling(positive: RDD[String],
                                                              negative: RDD[String],
                                                              htf : HashingTF,
                                                              sc:SparkContext,
                                                              k: Int,
                                                              sentiwordSet: util.HashSet[String]): RDD[LabeledPoint]  = {
    //    var negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    var positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    //    for( i <- 0 until 10){
    val positiveTemp = positive.union(blankout_generator(negative, k, sentiwordSet )).union(negative)
    //    }
    //    println("positives after blankout = " + positiveTemp.filter(x=> x.split(",")(1).equals("positive")).count())
    //    println("negatives after blankout = " + positiveTemp.filter(x=> x.split(",")(1).equals("negative")).count())

    val tempDataset = positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val positive_rdd = tempDataset .filter(_.label == 0).cache()
    val positive_rdd_count = positive_rdd.count()


    positive_rdd.union(sc.parallelize(tempDataset.filter( _.label == 1)
      .++(tempDataset .filter( _.label == 1))
      .++(tempDataset .filter( _.label == 1))
      .++(tempDataset .filter( _.label == 1))
      .take(positive_rdd_count.toInt)))  //.++(tempDataset .filter( _.label == 1)))

    //    println("Training set: positives after OverSampling = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives after OverSampling = " + finalDataset.filter(_.label == 1).count())

    //    cross_validation_function(finalDataset, original_test_evaluation)

  }


  def unbalance10foldWithBlankoutOnlyMinorityWithOverSamplingNoFilter(positive: RDD[String],
                                                                      negative: RDD[String],
                                                                      htf : HashingTF,
                                                                      sc:SparkContext,
                                                                      k: Int): RDD[LabeledPoint]  = {

    val positiveTemp = positive.union(blankout_generator_no_filter(negative, k)).union(negative)

    val tempDataset = positiveTemp.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val positive_rdd = tempDataset .filter(_.label == 0).cache()
    val positive_rdd_count = positive_rdd.count()


    positive_rdd.union(sc.parallelize(tempDataset.filter( _.label == 1)
      .++(tempDataset .filter( _.label == 1))
      .++(tempDataset .filter( _.label == 1))
      .++(tempDataset .filter( _.label == 1))
      .take(positive_rdd_count.toInt)))
  }

  def unbalance10foldWithBlankoutBothClassesWithOverSampling(positive: RDD[String],
                                                             negative: RDD[String],
                                                             htf : HashingTF,
                                                             sc:SparkContext,
                                                             k: Int,
                                                             sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    //    var negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    var positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))


    val blankGeneration = positive.union(blankout_generator(positive, k, sentiwordSet))
      .union(blankout_generator(negative, k , sentiwordSet))
      .union(negative)

    //    println("positives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("positive")).count())
    //    println("negatives after blankout = " + blankGeneration.filter(x=> x.split(",")(1).equals("negative")).count())


    val tempDataset = blankGeneration .map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val positive_rdd = tempDataset.filter(_.label == 0).cache()
    //    val positive_rdd_count = positive_rdd.count()

    positive_rdd.union(sc.parallelize(
      tempDataset .filter( _.label == 1)
        //        .++(tempDataset .filter( _.label == 1))
        //        .++(tempDataset .filter( _.label == 1))
        //        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .take(positive_rdd.count().toInt)))

    //    println("Training set: positives after OverSampling = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives after OverSampling = " + finalDataset.filter(_.label == 1).count())

    //    cross_validation_function(finalDataset, original_test_evaluation)
  }


  def unbalance10foldWithBlankoutBothClassesWithOverSamplingNoFilter(positive: RDD[String],
                                                                     negative: RDD[String],
                                                                     htf : HashingTF,
                                                                     sc:SparkContext,
                                                                     k: Int): RDD[LabeledPoint] = {
    val blankGeneration = positive.union(blankout_generator_no_filter(positive, k))
      .union(blankout_generator_no_filter(negative, k ))
      .union(negative)

    val tempDataset = blankGeneration .map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val positive_rdd = tempDataset.filter(_.label == 0).cache()

    positive_rdd.union(sc.parallelize(
      tempDataset .filter( _.label == 1)
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .++(tempDataset .filter( _.label == 1))
        .take(positive_rdd.count().toInt)))
  }

  def balanced10foldWithBlankout(rawDataset: RDD[String],
                                 htf : HashingTF,
                                 k: Int,
                                 sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {

    val tempset = blankout_generator(rawDataset, k, sentiwordSet)

    //    println("positives Blankout generation = " + tempset.filter(x=> x.split(",")(1).equals("positive")).count())
    //    println("negatives Blankout generation = " + tempset.filter(x=> x.split(",")(1).equals("negative")).count())

    tempset.union(rawDataset).map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

  }

  def unbalance10foldWithAntonyms(positive: RDD[String],
                                  negative:RDD[String],
                                  htf : HashingTF,
                                  wordnetMapper:mutable.HashMap[String,String],
                                  antonymous_case: Int): RDD[LabeledPoint] = {
    val unbalanced = positive.union(negative).cache

    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_function(unbalanced, htf, wordnetMapper, antonymous_case)

    dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
  }


  def unbalance10foldWithAntonymsFiltered(positive: RDD[String],
                                          negative:RDD[String],
                                          htf : HashingTF,
                                          wordnetMapper:mutable.HashMap[String,String],
                                          sentiwordSet: util.HashSet[String],
                                          antonymous_case: Int): RDD[LabeledPoint] = {
    val unbalanced = positive.union(negative).cache

    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_filtered_function(unbalanced, htf, wordnetMapper, sentiwordSet, antonymous_case)

    dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
  }


  def unbalance10foldWithAntonymsAndRebalanceUnderSampling(positive: RDD[String],
                                                           negative:RDD[String],
                                                           htf : HashingTF,
                                                           wordnetMapper:mutable.HashMap[String,String],
                                                           antonymous_case: Int,
                                                           sc:SparkContext): RDD[LabeledPoint] = {
    val unbalanced = positive.union(negative).cache

    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_function(unbalanced, htf, wordnetMapper, antonymous_case)

    val finalDatasetImbalanced = dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val negative_rdd = finalDatasetImbalanced.filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    negative_rdd.union(sc.parallelize(finalDatasetImbalanced.filter( _.label == 0).take(negative_count.toInt)))
  }



  def unbalance10foldWithAntonymsAndRebalanceUnderSamplingFiltered(positive: RDD[String],
                                                                   negative:RDD[String],
                                                                   htf : HashingTF,
                                                                   wordnetMapper:mutable.HashMap[String,String],
                                                                   sentiwordSet: util.HashSet[String],
                                                                   antonymous_case: Int,
                                                                   sc:SparkContext): RDD[LabeledPoint] = {
    val unbalanced = positive.union(negative).cache

    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_filtered_function(unbalanced, htf, wordnetMapper, sentiwordSet, antonymous_case)

    val finalDatasetImbalanced = dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val negative_rdd = finalDatasetImbalanced.filter(_.label == 1).cache()
    val negative_count = negative_rdd.count()

    negative_rdd.union(sc.parallelize(finalDatasetImbalanced.filter( _.label == 0).take(negative_count.toInt)))
  }



  def unbalance10foldWithAntonymsAndRebalanceOverSampling(positive: RDD[String],
                                                          negative:RDD[String],
                                                          htf : HashingTF,
                                                          wordnetMapper:mutable.HashMap[String,String],
                                                          antonymous_case: Int,
                                                          sc:SparkContext): RDD[LabeledPoint]= {
    val unbalanced = positive.union(negative).cache
    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_function(unbalanced, htf, wordnetMapper, antonymous_case)

    val finalDatasetImbalanced = dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val positive_rdd = finalDatasetImbalanced.filter(_.label == 0).cache()
    val positive_count = positive_rdd.count()

    positive_rdd.union(sc.parallelize(
      finalDatasetImbalanced.filter( _.label == 1)
        .++(finalDatasetImbalanced.filter( _.label == 1))
        .++(finalDatasetImbalanced.filter( _.label == 1))
        .++(finalDatasetImbalanced.filter( _.label == 1))
        .++(finalDatasetImbalanced.filter( _.label == 1))
        .take(positive_count.toInt)))
  }


  def unbalance10foldWithAntonymsAndRebalanceOverSamplingFiltered(positive: RDD[String],
                                                                  negative:RDD[String],
                                                                  htf : HashingTF,
                                                                  wordnetMapper:mutable.HashMap[String,String],
                                                                  sentiwordSet: util.HashSet[String],
                                                                  antonymous_case: Int,
                                                                  sc:SparkContext): RDD[LabeledPoint]= {
    val unbalanced = positive.union(negative).cache
    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class
    val dataSet = choose_antonymous_generation_filtered_function(unbalanced, htf, wordnetMapper, sentiwordSet, antonymous_case)

    val finalDatasetImbalanced = dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val positive_rdd = finalDatasetImbalanced.filter(_.label == 0).cache()
    val positive_count = positive_rdd.count()

    positive_rdd.union(sc.parallelize(
      finalDatasetImbalanced.filter( _.label == 1)
      .++(finalDatasetImbalanced.filter( _.label == 1))
      .++(finalDatasetImbalanced.filter( _.label == 1))
      .++(finalDatasetImbalanced.filter( _.label == 1))
      .++(finalDatasetImbalanced.filter( _.label == 1))
        .take(positive_count.toInt)))
  }



  def unbalance10foldWithAntonymsFairPerturbation(positive: RDD[String],
                                                  negative:RDD[String],
                                                  htf : HashingTF,
                                                  wordnetMapper:mutable.HashMap[String,String],
                                                  antonymous_case: Int,
                                                  sc:SparkContext): RDD[LabeledPoint]= {
    // filter to rdds
    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positive.union(negative).cache

    val positives_init = unbalanced.filter(x=> x.split(",")(1).equals("positive"))
    val negatives_init = unbalanced.filter(x=> x.split(",")(1).equals("negative"))

    val pos_init_count = positives_init.count()
    val neg_init_count = negatives_init.count()

    //    println("positives before antonym generation = " + pos_init_count )
    //    println("negatives before antonym generation = " + neg_init_count )

    // 0: paper based only imbalanced class,
    // 1: paper based for both classes,
    // 2: all negations for all classes,
    // 3: paper based for imbalance class only,
    // 4: only 1 word for both classes,
    // 5: only 1 word for imbalanced class


    val dataSet = choose_antonymous_generation_function(unbalanced, htf, wordnetMapper, antonymous_case)

    val messUpMajorityClass = negatives_init.union(

      sc.parallelize(positives_init.take(pos_init_count.toInt / 2))).map{
      line =>
        val parts = line.split(',')
        var id = 1.0
        if (parts(1).equals("positive")){
          id = 0.0
        }
        LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    val positive_rdd = dataSet.filter(_.label == 0).cache()
    val positive_count = positive_rdd.count()

    messUpMajorityClass.union(sc.parallelize(positive_rdd.filter( _.label == 0).take(pos_init_count.toInt / 2))
      .union(dataSet.filter(_.label == 1)))

    //    println("positives discarded and replaced with synthetic " + pos_init_count.toInt / 2)
    //    println("total positives final dataset antonym generation = " + finalDataset.filter(_.label == 0).count())
    //    println("total negatives final dataset antonym generation = " + finalDataset.filter(_.label == 1).count())

    //    cross_validation_function(finalDataset, original_test_evaluation)
  }
  //
  //  def unbalance10foldWithAntonymsSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF,
  //                                         wordnetMapper:mutable.HashMap[String,String]): RDD[(Double,Double)] = {
  //
  //    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
  //    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))
  //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
  //
  //    val unbalanced = positiveTemp.union(negative(1)).cache
  //    val dataSet = antonymsGeneratorSemEval(unbalanced, htf, wordnetMapper)
  //
  //    val model = NaiveBayes.train(dataSet.union(unbalanced.map{line =>
  //      val parts = line.split('\t')
  //      var id = 1.0
  //      if (parts(1).equals("positive")){
  //        id = 0.0
  //      }
  //      LabeledPoint(id , htf.transform(parts(3).split(' ')))
  //    }))
  //
  //    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  //      val prediction = model.predict(features)
  //      (prediction, label)
  //    }
  //    predictionAndLabels
  //  }


  def unbalance10foldWithSimilarsAllWordsAllClasses(positive: RDD[String],
                                                    negative:RDD[String],
                                                    htf : HashingTF,
                                                    mapper: mutable.HashMap[String, String],
                                                    sentiwordSet: util.HashSet[String],
                                                    sc:SparkContext,
                                                    percentage: Double): RDD[LabeledPoint] = {

    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))

    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorAllWordsAllClasses(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()

    sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
  }

  def unbalance10foldWithSimilarsAllWordsOnlyMinorityClass(positive: RDD[String],
                                                           negative:RDD[String],
                                                           htf : HashingTF,
                                                           mapper: mutable.HashMap[String, String],
                                                           sentiwordSet: util.HashSet[String],
                                                           sc:SparkContext,
                                                           percentage: Double): RDD[LabeledPoint]= {

    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorAllWordsOnlyMinorityClass(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()

    sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

  }

  def unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithUnderSampling(positive: RDD[String],
                                                                                     negative:RDD[String],
                                                                                     htf : HashingTF,
                                                                                     mapper: mutable.HashMap[String, String],
                                                                                     sc: SparkContext,
                                                                                     sentiwordSet: util.HashSet[String],
                                                                                     percentage: Double): RDD[LabeledPoint]= {

    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorAllWordsOnlyMinorityClass(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()


    val tempDataset = sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val neg_count = tempDataset.filter(_.label == 1).count()
    sc.parallelize(tempDataset.filter(_.label == 0).take(neg_count.toInt)).union(tempDataset.filter(_.label == 1)).cache()
 }

  def unbalance10foldWithSimilarsAllWordsOnlyMinorityClassRebalanceWithOverSampling(positive: RDD[String],
                                                                                    negative:RDD[String],
                                                                                    htf : HashingTF,
                                                                                    mapper: mutable.HashMap[String, String],
                                                                                    sc: SparkContext,
                                                                                    sentiwordSet: util.HashSet[String],
                                                                                    percentage: Double): RDD[LabeledPoint]= {


    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorAllWordsOnlyMinorityClass(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()

    val tempDataset = sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val negative_oversampling = tempDataset.filter(_.label == 1)
      .++(tempDataset.filter(_.label == 1))
      .++(tempDataset.filter(_.label == 1))
      .++(tempDataset.filter(_.label == 1))

    tempDataset.filter(_.label == 0).union(sc.parallelize(negative_oversampling.take(tempDataset.filter(_.label == 0).count.toInt))).cache()
  }

  def unbalance10foldWithSimilarsOnlyOneWordAllClasses(positive: RDD[String],
                                                       negative:RDD[String],
                                                       htf : HashingTF,
                                                       mapper: mutable.HashMap[String, String],
                                                       sentiwordSet: util.HashSet[String],
                                                       sc: SparkContext,
                                                       percentage: Double): RDD[LabeledPoint]= {

    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))

    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorOnlyOneWordAllClasses(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()

    sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
    //    println("Training set: positives before antonym generation = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives before antonym generation = " + finalDataset.filter(_.label == 1).count())


    //    cross_validation_function(finalDataset, original_test_evaluation)
  }

  def unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClass(positive: RDD[String],
                                                              negative:RDD[String],
                                                              htf : HashingTF,
                                                              mapper: mutable.HashMap[String, String],
                                                              sentiwordSet: util.HashSet[String],
                                                              sc: SparkContext,
                                                              percentage: Double): RDD[LabeledPoint]= {

    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorOnlyOneWordOnlyMinorityClass(unbalanced, htf, mapper, sentiwordSet)

    var dataset_count = (dataSet.count() * percentage).intValue()

    sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
    //    println("Training set: positives before antonym generation = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives before antonym generation = " + finalDataset.filter(_.label == 1).count())


    //    cross_validation_function(finalDataset, original_test_evaluation)
  }

  def unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithUnderSampling(positive: RDD[String],
                                                                                        negative:RDD[String],
                                                                                        htf : HashingTF,
                                                                                        mapper: mutable.HashMap[String, String],
                                                                                        sc:SparkContext,
                                                                                        sentiwordSet: util.HashSet[String],
                                                                                        percentage: Double): RDD[LabeledPoint]= {

    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorOnlyOneWordOnlyMinorityClass(unbalanced, htf, mapper,sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()
    val tempDataset = sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })
    val neg_count = tempDataset.filter(_.label == 1).count()

    sc.parallelize(tempDataset.filter(_.label == 0).take(neg_count.toInt)).union(tempDataset.filter(_.label == 1)).cache()
  }

  def unbalance10foldWithSimilarsOnlyOneWordOnlyMinorityClassRebalanceWithOverSampling(positive: RDD[String],
                                                                                       negative:RDD[String],
                                                                                       htf : HashingTF,
                                                                                       mapper: mutable.HashMap[String, String],
                                                                                       sc:SparkContext,
                                                                                       sentiwordSet: util.HashSet[String],
                                                                                       percentage: Double): RDD[LabeledPoint] = {

    val unbalanced = positive.union(negative).cache

    val dataSet = similarGeneratorOnlyOneWordOnlyMinorityClass(unbalanced, htf, mapper, sentiwordSet)
    var dataset_count = (dataSet.count() * percentage).intValue()

    val tempDataset = sc.parallelize(dataSet.take(dataset_count)).union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val negative_oversampling = tempDataset.filter(_.label == 1)
      .++(tempDataset.filter(_.label == 1))
      .++(tempDataset.filter(_.label == 1))
      .++(tempDataset.filter(_.label == 1))

    tempDataset.filter(_.label == 0).union(sc.parallelize(negative_oversampling.take(tempDataset.filter(_.label == 0).count.toInt))).cache()
  }

  def antonymsGeneratorPaperBasedOnlyImbalancedClass(data: RDD[String], htf: HashingTF,
                                                     wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if (wordnetMapper.contains(word)){
            negationExists = true
            unigramsAntonymous += wordnetMapper(word) + " "
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())

    antonyms.filter(_.label == 1)
  }

  def antonymsGeneratorFilteredPaperBasedOnlyImbalancedClass(data: RDD[String],
                                                             htf: HashingTF,
                                                             wordnetMapper:mutable.HashMap[String,String],
                                                             sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if (wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)){
            negationExists = true
            unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())

    antonyms.filter(_.label == 1)
  }

  def filter_antonyms_generation(data: RDD[String], wordnetMapper: mutable.HashMap[String, String]) : RDD[String] ={
    data.filter{ line =>
      val parts = line.split(",")

      var counter_negations = 0
      for (word <- parts(2).split(" ")) {
        if (word.toLowerCase.startsWith("not")) {
          counter_negations += 1
        } else if(word.toLowerCase.endsWith("not")) {
          counter_negations += 1
        }
      }

      var counter_antonyms = 0
      for (word <- parts(2).split(" ")) {
        if (wordnetMapper.contains(word.toLowerCase)) {
          counter_antonyms += 1
        }
      }
      !((counter_antonyms >= 1 && counter_negations >= 1) || counter_negations >= 2 || counter_antonyms >= 2)
    }
  }

  def antonymsGeneratorPaperBasedBothClasses(data: RDD[String], htf: HashingTF,
                                             wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {

    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.toLowerCase.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.toLowerCase.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if (wordnetMapper.contains(word.toLowerCase)){
            negationExists = true
            unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())

    //    denoise_synthetic_data(antonyms)
    antonyms
  }

  def antonymsGeneratorFilteredPaperBasedBothClasses(data: RDD[String],
                                                     htf: HashingTF,
                                                     wordnetMapper:mutable.HashMap[String,String],
                                                     sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {

    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.toLowerCase.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.toLowerCase.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if (wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)){
            negationExists = true
            unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())

    //    denoise_synthetic_data(antonyms)
    antonyms
  }

  def antonymsGeneratorNegationsAndWordsBothClasses(data: RDD[String], htf: HashingTF,
                                                    wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {

    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        } else if(wordnetMapper.contains(word)){
          negationExists = true
          unigramsAntonymous += wordnetMapper(word) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms
  }

  def antonymsGeneratorFilteredNegationsAndWordsBothClasses(data: RDD[String], htf: HashingTF,
                                                            wordnetMapper:mutable.HashMap[String,String],
                                                            sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {

    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        } else if(wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)){
          negationExists = true
          unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms
  }


  def antonymsGeneratorNegationsAndWordsOnlyImbalancedClass(data: RDD[String], htf: HashingTF,
                                                            wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        } else if(wordnetMapper.contains(word)){
          negationExists = true
          unigramsAntonymous += wordnetMapper(word) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }


  def antonymsGeneratorFilteredNegationsAndWordsOnlyImbalancedClass(data: RDD[String], htf: HashingTF,
                                                                    wordnetMapper:mutable.HashMap[String,String],
                                                                    sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        } else if(wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)){
          negationExists = true
          unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }

  def antonymsGeneratorOnlyOneWordOnlyImbalancedClass(data: RDD[String], htf: HashingTF,
                                                      wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if(!negationExists) {
          if (word.startsWith("NOT")) {
            negationExists = true
            unigramsAntonymous += word.replace("NOT", "") + " "
          } else if (word.endsWith("not")) {
            negationExists = true
            unigramsAntonymous += word.substring(0, word.length - 3) + " "
          }
          else {
            unigramsAntonymous += word + " "
          }
        }else{
          unigramsAntonymous += word + " "
        }

      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if(!negationExists) {
            if (wordnetMapper.contains(word)) {
              negationExists = true
              unigramsAntonymous += wordnetMapper(word) + " "
            } else {
              unigramsAntonymous += word + " "
            }
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }

  def antonymsGeneratorFilteredOnlyOneWordOnlyImbalancedClass(data: RDD[String], htf: HashingTF,
                                                              wordnetMapper:mutable.HashMap[String,String],
                                                              sentiwordSet: util.HashSet[String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if(!negationExists) {
          if (word.startsWith("NOT")) {
            negationExists = true
            unigramsAntonymous += word.replace("NOT", "") + " "
          } else if (word.endsWith("not")) {
            negationExists = true
            unigramsAntonymous += word.substring(0, word.length - 3) + " "
          }
          else {
            unigramsAntonymous += word + " "
          }
        }else{
          unigramsAntonymous += word + " "
        }

      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if(!negationExists) {
            if (wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)) {
              negationExists = true
              unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
            } else {
              unigramsAntonymous += word + " "
            }
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }

  def antonymsGeneratorOnlyOneWordBothClasses(data: RDD[String], htf: HashingTF,
                                              wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if(!negationExists) {
          if (word.startsWith("NOT")) {
            negationExists = true
            unigramsAntonymous += word.replace("NOT", "") + " "
          } else if (word.endsWith("not")) {
            negationExists = true
            unigramsAntonymous += word.substring(0, word.length - 3) + " "
          }
          else {
            unigramsAntonymous += word + " "
          }
        }else{
          unigramsAntonymous += word + " "
        }

      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if(!negationExists) {
            if (wordnetMapper.contains(word)) {
              negationExists = true
              unigramsAntonymous += wordnetMapper(word) + " "
            } else {
              unigramsAntonymous += word + " "
            }
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms
  }

  def antonymsGeneratorFilteredOnlyOneWordBothClasses(data: RDD[String], htf: HashingTF,
                                                      wordnetMapper:mutable.HashMap[String,String],
                                                      sentiwordSet: util.HashSet[String] ): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(2).split(" ")) {
        if(!negationExists) {
          if (word.startsWith("NOT")) {
            negationExists = true
            unigramsAntonymous += word.replace("NOT", "") + " "
          } else if (word.endsWith("not")) {
            negationExists = true
            unigramsAntonymous += word.substring(0, word.length - 3) + " "
          }
          else {
            unigramsAntonymous += word + " "
          }
        }else{
          unigramsAntonymous += word + " "
        }

      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(2).split(" ")) {
          if(!negationExists) {
            if (wordnetMapper.contains(word.toLowerCase) && sentiwordSet.contains(word.toLowerCase)) {
              negationExists = true
              unigramsAntonymous += wordnetMapper(word.toLowerCase) + " "
            } else {
              unigramsAntonymous += word + " "
            }
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "," + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split(',')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("positives antonym generated = " + antonyms.filter(_.label == 0).count())
    //    println("negatives antonym generated = " + antonyms.filter(_.label == 1).count())
    antonyms
  }

  def antonymsGeneratorSemEval(data: RDD[String], htf: HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
    val antonyms = data.map { line =>
      val parts = line.split("\t")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(3).split(" ")) {
        if (word.startsWith("NOT")) {
          negationExists = true
          unigramsAntonymous += word.replace("NOT", "") + " "
        } else if(word.endsWith("not")) {
          negationExists = true
          unigramsAntonymous += word.substring(0, word.length - 3) + " "
        }
        else{
          unigramsAntonymous += word + " "
        }
      }
      if (!negationExists){
        unigramsAntonymous = ""
        for (word <- parts(3).split(" ")) {
          if (wordnetMapper.contains(word)){
            negationExists = true
            unigramsAntonymous += wordnetMapper(word) + " "
          }else{
            unigramsAntonymous += word + " "
          }
        }
      }

      if(negationExists){
        var status = "positive"
        if (parts(1).equals(status)){
          status = "negative"
        }
        status + "\t" + unigramsAntonymous.trim
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
      .map{
        line =>
          val parts = line.split('\t')
          var id = 1.0
          if (parts(0).equals("positive")){
            id = 0.0
          }
          LabeledPoint(id , htf.transform(parts(1).split(' ')))
      }
    //    println("antonyms generated = " + antonyms.count())
    antonyms
  }


  def callBinaryMetricsFunction(predictionAndLabels: RDD[(Double, Double)], output: String): BinaryClassificationMetrics ={
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    // Precision by threshold
    val precision = metrics.precisionByThreshold.collectAsMap()
    println(" Precision for class positive: " + precision.get(0.0))
    println(" Precision for class negative: " + precision.get(1.0))


    // Recall by threshold
    val recall = metrics.recallByThreshold().collectAsMap()
    /*
        recall.foreach { case (t, r) =>
          println("Threshold:" + t+ " Recall: " + r)
        }
    */
    println(" Recall for class positive: " + recall.get(0.0))
    println(" Recall for class negative: " + recall.get(1.0))

    // Precision-Recall Curve
    //    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold.collectAsMap()

    println(" f1Score for class positive: " + f1Score.get(0.0))
    println(" f1Score for class negative: " + f1Score.get(1.0))

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    //    val thresholds = precision.keys

    // ROC Curve
    //    val roc = metrics.roc.saveAsTextFile(output)

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
    metrics
  }

  def callMultiMetricsFunction(predictionAndLabels: RDD[(Double, Double)]): Unit = {
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }
    // F-measure by label
    labels.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // Weighted stats
    println(s"Weighted precision: ${metrics.weightedPrecision}")
    println(s"Weighted recall: ${metrics.weightedRecall}")
    println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
    println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
  }

  def unbalance10foldWithOversampling(positive: RDD[String],
                                      negative:RDD[String],
                                      htf : HashingTF,
                                      sc: SparkContext): RDD[LabeledPoint]= {

    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val oversamplingSet = negative.++(negative).++(negative).++(negative).++(negative).++(negative).++(negative).++(negative).++(negative).++(negative).cache
    //    println("negative(1).count : " + negative.count )
    //    println("oversamplingSet.count : " + oversamplingSet.count)
    val unbalanced = positive.union(sc.parallelize(oversamplingSet.take(positive.count.toInt))).cache

    unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }

    //    println("Training set: positives before antonym generation = " + finalDataset.filter(_.label == 0).count())
    //    println("Training set: negatives before antonym generation = " + finalDataset.filter(_.label == 1).count())
    //    cross_validation_function(finalDataset, original_test_evaluation)

  }


  def unbalance10foldWithUndersampling(positive: RDD[String],
                                       negative:RDD[String],
                                       htf : HashingTF,
                                       sc: SparkContext): RDD[LabeledPoint]= {

    //    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    //    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    //    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val negativeSize = negative.count()
    //    println("negativeSize and positiveSize equal size = " + negativeSize )
    //    sc.parallelize(positiveTemp.take(negativeSize.toInt))

    val dataset= sc.parallelize(positive.take(negativeSize.toInt)).union(negative).cache
    dataset.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }
  }

}
