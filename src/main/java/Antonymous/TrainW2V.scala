package Antonymous

/**
  * Created by iosifidis on 19.09.16.
  */
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}

object TrainW2V {



  def main(args: Array[String]) {
//    val myModel : Word2VecLoader = new Word2VecLoader
    def conf = new SparkConf().setAppName(this.getClass.getName)
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
//    val data = sc.textFile(args(0)).filter(x=> x.split(",")(1).equals("negative")).cache
    val input = sc.textFile(args(0)).map(line => line.split(" ").toSeq).cache

    val word2vec = new Word2Vec()
    val model = word2vec.fit(input)

    println(model.findSynonyms("ugly", 5))
    println(model.findSynonyms("funny", 5))
    model.save(sc, path="word2vec_stored_model")
  }
}

