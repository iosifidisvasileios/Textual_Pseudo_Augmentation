package Antonymous

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.HashMap

/**
  * Created by iosifidis on 07.07.17.
  */
class SyntheticGenerationFunctions {

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

  def similarGenerator(data: RDD[String], htf: HashingTF, mapper : mutable.HashMap[String, String]): RDD[LabeledPoint] = {
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
  def similarGeneratorSemEval(data: RDD[String], htf: HashingTF, mapper : mutable.HashMap[String, String]): RDD[LabeledPoint] = {
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

  def unbalance10foldWithAntonyms(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[(Double,Double)] = {
    // filter to rdds
    val negativeTemp = rawDataset.filter(x=> x.split(",")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split(",")(1).equals("positive"))
    // create imbalance to negative class
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))
    val unbalanced = positiveTemp.union(negative(1)).cache


    println("positives before antonym generation = " + unbalanced.filter(x=> x.split(",")(1).equals("positive")).count())
    println("negatives before antonym generation = " + unbalanced.filter(x=> x.split(",")(1).equals("negative")).count())

    // generate antonymous
//    val dataSet = antonymsGenerator(unbalanced, htf, wordnetMapper)
    val dataSet = antonymsGeneratorOnlyOneWordOnlyImbalancedClass(unbalanced, htf, wordnetMapper)
    val finalDataset = dataSet.union(unbalanced.map{line =>
      val parts = line.split(',')
      var id = 1.0
      if (parts(1).equals("positive")){
        id = 0.0
      }
      LabeledPoint(id , htf.transform(parts(2).split(' ')))
    })

    println("positives after antonym generation = " + finalDataset.filter(_.label == 0).count())
    println("negatives after antonym generation = " + finalDataset.filter(_.label == 1).count())
    val model = NaiveBayes.train(finalDataset)

    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels
  }

  def unbalance10foldWithAntonymsSemEval(rawDataset: RDD[String], test: RDD[LabeledPoint], htf : HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[(Double,Double)] = {

    val negativeTemp = rawDataset.filter(x=> x.split("\t")(1).equals("negative"))
    val positiveTemp = rawDataset.filter(x=> x.split("\t")(1).equals("positive"))
    val negative = negativeTemp.randomSplit(Array(0.9, 0.1 ))

    val unbalanced = positiveTemp.union(negative(1)).cache
    val dataSet = antonymsGeneratorSemEval(unbalanced, htf, wordnetMapper)

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


  def antonymsGenerator(data: RDD[String], htf: HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
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
    println("positives after antonym generation = " + antonyms.filter(_.label == 0).count())
    println("negatives after antonym generation = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }

  def antonymsGeneratorOnlyOneWordOnlyImbalancedClass(data: RDD[String], htf: HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
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
    println("antonyms generated = " + antonyms.count())
    println("positives after antonym generation = " + antonyms.filter(_.label == 0).count())
    println("negatives after antonym generation = " + antonyms.filter(_.label == 1).count())
    antonyms.filter(_.label == 1)
  }
  def antonymsGeneratorOnlyOneWordBothClasses(data: RDD[String], htf: HashingTF, wordnetMapper:mutable.HashMap[String,String]): RDD[LabeledPoint] = {
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
    println("antonyms generated = " + antonyms.count())
    println("positives after antonym generation = " + antonyms.filter(_.label == 0).count())
    println("negatives after antonym generation = " + antonyms.filter(_.label == 1).count())
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


}