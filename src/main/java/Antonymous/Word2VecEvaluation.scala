//package Antonymous
//
///**
//  * Created by iosifidis on 19.09.16.
//  */
//import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
//
//object Word2VecEvaluation {
//
//
//
//  def main(args: Array[String]) {
//    val myModel : Word2VecLoader = new Word2VecLoader
//    def conf = new SparkConf().setAppName(this.getClass.getName)
//    val sc = new SparkContext(conf)
//    sc.setLogLevel("ERROR")
//    val data = sc.textFile(args(0)).filter(x=> x.split(",")(1).equals("negative")).cache
//
////    println( "negative: " + data.count() )
//    val dataMapped = data.map{ line =>
//      val text = line.split(",")(2)
//      var reconstructed = ""
//      var w2vUsed = false
//      for (word <- text.split(" ")){
//        try {
//          val similarWord = myModel.findSimilarWord(word)
//          reconstructed += similarWord + " "
//          w2vUsed = true
//        }catch{
//          case _ => reconstructed += word + " "
//        }
//      }
//      (reconstructed, w2vUsed)
//    }
//
//    println("dataMapped : " + dataMapped.filter(_._2).count())
//
//  }
//
//}
//
