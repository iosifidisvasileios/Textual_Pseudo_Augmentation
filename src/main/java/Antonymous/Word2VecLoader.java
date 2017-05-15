//package Antonymous;
//
//
//import org.apache.log4j.Logger;
//import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
//import org.deeplearning4j.models.word2vec.Word2Vec;
//
//import java.io.File;
//import java.io.Serializable;
//import java.util.Collection;
//
///**
// * Created by iosifidis on 09.05.17.
// */
//public class Word2VecLoader implements Serializable {
//    private Word2Vec vec;
//    public Word2VecLoader() {
//        initializeModel();
//    }
//    static Logger logger = Logger.getLogger(Word2VecLoader.class);
//
//    private void initializeModel() {
////         File gModel = new File("/home/iosifidis/Downloads/GoogleNews-vectors-negative300.bin.gz");
//        File gModel = new File("GoogleNews-vectors-negative300.bin.gz");
//        try {
//            vec = WordVectorSerializer.readWord2VecModel(gModel);
//        }catch(Exception e){
//            logger.error(e ,e );
//            System.exit(1);
//        }
//    }
//
//    public String findSimilarWord (String s){
//        Collection<String> processed = vec.wordsNearest(s, 1);
//        return processed.iterator().next();
//    }
//
//}
