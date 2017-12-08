import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.io.Source

object GenerateAlvaroDataset {

  val syntheticGenerationFunctions = new SyntheticGenerationFunctions
  val word2vecFromGoogleMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromTwitterMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromGoogleSentimentMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromTwitterSentimentMapper = new mutable.HashMap[String,String]()  { }
  val word2vecFromDatasetMapper = new mutable.HashMap[String,String]()  { }
  val wordnetMapper  = new mutable.HashMap[String,String]()  { }


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
    val directory = args(1)

    // assign words to numbers converter
    val htf = new HashingTF(1500000)


    //----------------------Sentiment140-------------------------------
    data.union(antonymsGeneratorOnlyOneWordBothClasses(data, htf, wordnetMapper)).saveAsTextFile(directory + "/antonymous_dataset")

    data.union(similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleMapper)).saveAsTextFile(directory + "/google_generic")
    data.union(similarGeneratorAllWordsAllClasses(data, htf, word2vecFromGoogleSentimentMapper)).saveAsTextFile(directory + "/google_sentiment")

    data.union(similarGeneratorAllWordsAllClasses(data, htf, word2vecFromTwitterMapper)).saveAsTextFile(directory + "/twitter_generic")
    data.union(similarGeneratorAllWordsAllClasses(data, htf, word2vecFromTwitterSentimentMapper)).saveAsTextFile(directory + "/twitter_sentiment")

    data.union(blankout_generator(data)).saveAsTextFile(directory + "/blankout")

  }


  def antonymsGeneratorOnlyOneWordBothClasses_tsentiment15(data: RDD[String], htf: HashingTF,
                                              wordnetMapper:mutable.HashMap[String,String]): RDD[String] = {
    val antonyms = data.map { line =>
      val parts = line.split(",")
      var unigramsAntonymous = ""
      var negationExists = false

      for (word <- parts(3).split(" ")) {
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
        for (word <- parts(3).split(" ")) {
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
        var status = "1"
        if (parts(4).trim().equals("1")){
          status = "0"
        }else if(parts(4).trim().equals("0")){
          status = "1"
        }
         parts(0) + "," + parts(1) + "," + parts(2) + "," + unigramsAntonymous.trim() + "," + status

      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
    antonyms
  }

  def blankout_generator_tsentiment15(negatives: RDD[String]): RDD[String] = {
    val r = new java.util.Random
    negatives.filter(x => x.split(",")(3).split(" ").length >= 4).map{ line=>
      val parts = line.split(',')
      val number_of_words = parts(3).split(' ').length
      var tempString = ""
      val random_dropout = r.nextInt(number_of_words)
      var counter = 0

      for (word <- parts(3).split(" ")){
        if (counter != random_dropout){
          tempString += word + " "
        }
        counter += 1
      }
      parts(0) + "," + parts(1) + "," + parts(2) + "," + tempString.trim() + "," + parts(4).trim()
    }

  }


  def similarGeneratorAllWordsAllClasses_tsentiment15(data: RDD[String], htf: HashingTF, mapper : mutable.HashMap[String, String]): RDD[String] = {
    //    println("size of dictionary : " + mapper.size)

    val synonymous = data.map { line =>
      val parts2 = line.split(",")
      val line2 = line.toLowerCase

      val text = line2.split(",")(3)
      var unigramsSynonymous = ""
      var similarityFlag = false

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
        parts2(0) + "," + parts2(1) + "," + parts2(2) + "," + unigramsSynonymous.trim() + "," + parts2(4)
      }else{
        "-1"
      }
    }.filter(x=> x != "-1")
    synonymous
  }



  def blankout_generator(negatives: RDD[String]): RDD[String] = {
    val r = new java.util.Random
    negatives.filter(x => x.split(",")(2).split(" ").length >= 4).map{ line=>
      val parts = line.split(',')
      val number_of_words = parts(2).split(' ').length
      var tempString = ""
      val random_dropout = r.nextInt(number_of_words)
      var counter = 0

      for (word <- parts(2).split(" ")){
        if (counter != random_dropout){
          tempString += word + " "
        }
        counter += 1
      }
      parts(0) + "," + parts(1) + "," + tempString.trim()
    }

  }


  def similarGeneratorAllWordsAllClasses(data: RDD[String], htf: HashingTF, mapper : mutable.HashMap[String, String]): RDD[String] = {
    //    println("size of dictionary : " + mapper.size)

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
    synonymous
  }


  def antonymsGeneratorOnlyOneWordBothClasses(data: RDD[String], htf: HashingTF,
                                              wordnetMapper:mutable.HashMap[String,String]): RDD[String] = {
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
    antonyms
  }

}

