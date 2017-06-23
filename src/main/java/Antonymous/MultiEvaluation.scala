package Antonymous

/**
  * Created by iosifidis on 19.09.16.
  */

import collection.mutable.HashMap

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object MultiEvaluation {

  val word2vecFromGoogleMapper = new HashMap[String,String]()  { }
  val word2vecFromDatasetMapper = new HashMap[String,String]()  { }
  val wordnetMapper  = new HashMap[String,String]()  { }

  def simple10fold(data: RDD[LabeledPoint], test: RDD[LabeledPoint]): RDD[(Double,Double)] = {

    val model = NaiveBayes.train(data)
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10fold(data: RDD[LabeledPoint], test: RDD[LabeledPoint]): RDD[(Double,Double)] = {

    val negativeTemp = data.filter(x=> x.label.equals(1.0))
    val positiveTemp = data.filter(x=> x.label.equals(0.0))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val dataSet = positiveTemp.union(negative(1)).cache
    val model = NaiveBayes.train(dataSet)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def similarGenerator(data: RDD[String], htf: HashingTF, mapper : HashMap[String, String]): RDD[LabeledPoint] = {
    println("size of dictionary : " + mapper.size)

    val synonymous = data.map { line =>
      val line2 = line.toLowerCase
      val parts = line2.split(",")
      val text = line2.split(",")(2)
      var unigramsSynonymous = ""
      var similarityFlag = false
      val status = parts(1)

      for (word <- text.split(" ")){
        if (mapper.contains(word)){
          similarityFlag = true
          println("it is true! " + mapper(word))
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
    println("instances = " + synonymous.count())
    synonymous
  }
  def similarGeneratorSemEval(data: RDD[String], htf: HashingTF, mapper : HashMap[String, String]): RDD[LabeledPoint] = {
    println("size of dictionary : " + mapper.size)

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
          println("it is true! " + mapper(word))
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
    println("instances = " + synonymous.count())
    synonymous
  }

  def unbalance10foldWithAntonyms(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positiveTemp.union(negative(1)).cache
    val dataSet = antonymsGenerator(unbalanced, htf)

    val model = NaiveBayes.train(dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }))

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10foldWithAntonymsSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val unbalanced = positiveTemp.union(negative(1)).cache
    val dataSet = antonymsGeneratorSemEval(unbalanced, htf)

    val model = NaiveBayes.train(dataSet.union(unbalanced.map{line =>
      val parts = line.split('\t')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(3).split(' ')))
    }))

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }


  def unbalance10foldWithSimilars(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, mapper: HashMap[String, String]): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))

    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positiveTemp.union(negative(1)).cache

    val dataSet = similarGenerator(unbalanced, htf, mapper)


    val model = NaiveBayes.train(dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }))

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10foldWithSimilarsSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, mapper: HashMap[String, String]): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))

    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positiveTemp.union(negative(1)).cache

    val dataSet = similarGeneratorSemEval(unbalanced, htf, mapper)


    val model = NaiveBayes.train(dataSet.union(unbalanced.map{line =>
      val parts = line.split('\t')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(3).split(' ')))
    }))

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }


  def antonymsGenerator(data: RDD[String], htf: HashingTF): RDD[LabeledPoint] = {
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
    println("antonyms generated = " + antonyms.count())
    antonyms
  }

  def antonymsGeneratorSemEval(data: RDD[String], htf: HashingTF): RDD[LabeledPoint] = {
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
    println("antonyms generated = " + antonyms.count())
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
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold.collectAsMap()

    println(" f1Score for class positive: " + f1Score.get(0.0))
    println(" f1Score for class negative: " + f1Score.get(1.0))

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

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

  def unbalance10foldWithOversampling(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val oversamplingSet = negative(1).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).cache
    println("negative(1).count : " + negative(1).count )
    println("oversamplingSet.count : " + oversamplingSet.count)
    val unbalanced = positiveTemp.union(oversamplingSet).cache

    val model = NaiveBayes.train(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }


  def unbalance10foldWithOversamplingSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.8, 0.2 ))

    val oversamplingSet = negative(1).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).++(negative(1)).cache
    println("negative(1).count : " + negative(1).count )
    println("oversamplingSet.count : " + oversamplingSet.count)
    val unbalanced = positiveTemp.union(oversamplingSet).cache

    val model = NaiveBayes.train(unbalanced.map{line =>
      val parts = line.split('\t')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(3).split(' ')))
    })

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10foldWithUndersampling(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, sc: SparkContext): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val negativeSize = negative(1).count()
    println("negativeSize and positiveSize equal size = " + negativeSize )
    //    sc.parallelize(positiveTemp.take(negativeSize.toInt))

    val dataset= sc.parallelize(positiveTemp.take(negativeSize.toInt)).union(negative(1)).cache

    val model = NaiveBayes.train(dataset.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10foldWithUndersamplingSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, sc: SparkContext): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val negativeSize = negative(1).count()
    println("negativeSize and positiveSize equal size = " + negativeSize )
    //    sc.parallelize(positiveTemp.take(negativeSize.toInt))

    val dataset= sc.parallelize(positiveTemp.take(negativeSize.toInt)).union(negative(1)).cache

    val model = NaiveBayes.train(dataset.map{line =>
      val parts = line.split('\t')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(3).split(' ')))
    })

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }





  def main(args: Array[String]) {
    def conf = new SparkConf().setAppName(this.getClass.getName)
    for (line <- Source.fromFile("wordnet_antonymous.txt").getLines) {
      val words =line.split(",")
      wordnetMapper += (words(0) -> words(1))
    }

    for (line <- Source.fromFile("similar_words.txt").getLines) {
      val words = line.split("\t")
      word2vecFromGoogleMapper += (words(0) -> words(1).split(",")(0))
    }
    for (line <- Source.fromFile("similar_words_based_on_distant_supervision_trained_model.txt").getLines) {
      val words = line.split("\t")
      word2vecFromDatasetMapper += (words(0) -> words(1).split(",")(0))
    }

    val sc = new SparkContext(conf)
    println("Hello, world!")
    sc.setLogLevel("ERROR")
    val data = sc.textFile(args(0))
    val semEvalData = sc.textFile(args(1))
    val htf = new HashingTF(1500000)
    val htf_2 = new HashingTF(1500000)

    val dataSet = data.map { line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    }.cache()

    val dataSplit = dataSet.randomSplit(Array(0.7,0.3))

    //-----------------------SemEval----------------------------------
    val dataSetSemEval = semEvalData.map { line =>
      val parts = line.split('\t')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf_2.transform(parts(3).split(' ')))
    }.cache()
    println("------------------------------------------------------------")
    println("positive: " + dataSetSemEval.filter(_.label == 0.0).count)
    println("negative: " + dataSetSemEval.filter(_.label == 1.0).count)
    println("------------------------------------------------------------")
    val temp_balanced = dataSetSemEval.filter(_.label == 1.0).union(sc.parallelize(dataSetSemEval.filter(_.label == 0.0).take(11427)))


//    val dataSplitSemEval = dataSetSemEval.randomSplit(Array(0.7,0.3))
    val dataSplitSemEval = temp_balanced.randomSplit(Array(0.7,0.3))
    //---------------------------------------------------------



    //    println("Simple Balanced Scenario            ---------------------------------------------------")
    //    callBinaryMetricsFunction(simple10fold(dataSplit(0), dataSplit(1)),"simple_10_Fold")//.scoreAndLabels.saveAsTextFile("simple_10_Fold")

    //    println("Balanced Scenario with Antonymous   ---------------------------------------------------")
    //    callBinaryMetricsFunction(simple10fold(dataSplit(0).union(antonymsGenerator(data, htf)), dataSplit(1)), "simple_10_Fold_With_antonymous")
    //
    //    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10fold(dataSplit(0), dataSplit(1)), "simple_imbalanced")
    //
    //    println("Imbalanced Scenario with Antonymous ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10foldWithAntonyms(data, dataSplit(1), htf), "simple_imbalanced_with_antonymous")

    //    println("Imbalanced Scenario with OverSampling ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10foldWithOversampling(data, dataSplit(1), htf), "simple_imbalanced_with_oversampling")

    //    println("Imbalanced Scenario with UnderSampling ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10foldWithUndersampling(data, dataSplit(1), htf, sc), "simple_imbalanced_with_undersampling")

    //
    //    println("Balanced Synonymous with Google     ---------------------------------------------------")
    //    callBinaryMetricsFunction(simple10fold(dataSplit(0).union(similarGenerator(data, htf, word2vecFromGoogleMapper)), dataSplit(1)), "simple_10_Fold_With_Synonymous")

    //    println("Imbalanced Synonymous with Google   ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10foldWithSimilars(data, dataSplit(1), htf, word2vecFromGoogleMapper), "simple_imbalanced_With_Synonymous")

    //    println("Balanced Synonymous with Dataset    ---------------------------------------------------")
    //    callBinaryMetricsFunction(simple10fold(dataSplit(0).union(similarGenerator(data, htf, word2vecFromDatasetMapper)), dataSplit(1)), "simple_10_Fold_With_Synonymous")

    //    println("Imbalanced Synonymous with Dataset  ---------------------------------------------------")
    //    callBinaryMetricsFunction(unbalance10foldWithSimilars(data, dataSplit(1), htf, word2vecFromDatasetMapper), "simple_imbalanced_With_Synonymous")




    //-------------------------------------_SemEval-----------------------------------------------------
    println("Simple Balanced Scenario SemEval      ---------------------------------------------------")
    callBinaryMetricsFunction(simple10fold(dataSplitSemEval(0), dataSplitSemEval(1)),"simple_10_Fold")//.scoreAndLabels.saveAsTextFile("simple_10_Fold")


    println("Balanced Scenario with Antonymous   ---------------------------------------------------")
    callBinaryMetricsFunction(simple10fold(dataSplitSemEval(0).union(antonymsGeneratorSemEval(semEvalData, htf_2)), dataSplitSemEval(1)), "simple_10_Fold_With_antonymous")

    println("Simple Imbalanced Scenario          ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10fold(dataSplitSemEval(0), dataSplitSemEval(1)), "simple_imbalanced")

    println("Imbalanced Scenario with Antonymous ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10foldWithAntonymsSemEval(semEvalData, dataSplitSemEval(1), htf_2), "simple_imbalanced_with_antonymous")

    println("Imbalanced Scenario with OverSampling Sem EVAL  ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10foldWithOversamplingSemEval(semEvalData, dataSplitSemEval(1), htf_2), "simple_imbalanced_with_oversampling")

    println("Imbalanced Scenario with UnderSampling Sem EVAL ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10foldWithUndersamplingSemEval(semEvalData, dataSplitSemEval(1), htf_2, sc), "simple_imbalanced_with_undersampling")


    println("Balanced Synonymous with Google     ---------------------------------------------------")
    callBinaryMetricsFunction(simple10fold(dataSplitSemEval(0).union(similarGeneratorSemEval(semEvalData, htf_2, word2vecFromGoogleMapper)), dataSplitSemEval(1)), "simple_10_Fold_With_Synonymous")
    println("Balanced Synonymous with Dataset    ---------------------------------------------------")
    callBinaryMetricsFunction(simple10fold(dataSplitSemEval(0).union(similarGeneratorSemEval(semEvalData, htf_2, word2vecFromDatasetMapper)), dataSplitSemEval(1)), "simple_10_Fold_With_Synonymous")

    println("Imbalanced Synonymous with Google   ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10foldWithSimilarsSemEval(semEvalData, dataSplitSemEval(1), htf_2, word2vecFromGoogleMapper), "simple_imbalanced_With_Synonymous")
    println("Imbalanced Synonymous with Dataset  ---------------------------------------------------")
    callBinaryMetricsFunction(unbalance10foldWithSimilarsSemEval(semEvalData, dataSplitSemEval(1), htf_2, word2vecFromDatasetMapper), "simple_imbalanced_With_Synonymous")

  }
}

