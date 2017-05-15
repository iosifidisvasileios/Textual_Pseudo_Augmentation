//package Antonymous; /**
// * Created by iosifidis on 08.08.16.
// */
//
//import org.apache.hadoop.conf.Configuration;
//import org.apache.hadoop.fs.Path;
//import org.apache.hadoop.io.Text;
//import org.apache.hadoop.mapreduce.Job;
//import org.apache.hadoop.mapreduce.Mapper;
//import org.apache.hadoop.mapreduce.Reducer;
//import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
//import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.apache.log4j.Logger;
//
//import java.io.IOException;
//import java.net.URI;
//
//public class ParallelAntonymousCreation {
//
//    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
//        static Logger logger = Logger.getLogger(ParallelAntonymousCreation.class);
//        private static Word2VecLoader word2VecLoader;
//
//        @Override
//        protected void setup(Mapper.Context context) throws IOException, InterruptedException {
//            logger.info("initialize word2vec model");
//            word2VecLoader = new Word2VecLoader();
//            logger.info("word2vec model loaded");
//        }
//
//        public void map(Object key, Text row, Context context) throws IOException, InterruptedException {
//            final String items = row.toString();
//            String[] stringArray = items.split(",");
//            String costruction = "";
//
//            for (String word : stringArray[2].split(" ")){
//                try{
//                    costruction += word2VecLoader.findSimilarWord(word) + " ";
//                }catch(Exception e){
//                }
//            }
//            context.write(new Text(stringArray[0]), new Text(stringArray[1] + "," + stringArray[2] + "," + costruction));
//
//        }
//    }
//
//    static class MyReducer extends Reducer<Text, Text, Text, Text>{
//        protected void reduce(Text key, Text values, Context context) throws IOException, InterruptedException {
//            context.write(key, new Text(values));
//        }
//    }
//
//    public static void main(String[] args) throws Exception {
//        Configuration conf = new Configuration();
//
//        conf.set("mapreduce.map.memory.mb", "16096");
//        conf.set("mapreduce.map.java.opts", "-Xmx16096m");
//        Job job = Job.getInstance(conf, ParallelAntonymousCreation.class.getName());
//        job.addCacheFile(new URI("hdfs://nameservice1/user/iosifidis/GoogleNews-vectors-negative300.bin.gz#GoogleNews-vectors-negative300.bin.gz"));
//
//
//        job.setJarByClass(ParallelAntonymousCreation.class);
//        job.setMapperClass(TokenizerMapper.class);
//
//        job.setCombinerClass(MyReducer.class);
//        job.setReducerClass(MyReducer.class);
//
//        job.setOutputKeyClass(Text.class);
//        job.setOutputValueClass(Text.class);
//
//        FileInputFormat.addInputPath(job, new Path(args[0]));
//        FileOutputFormat.setOutputPath(job, new Path(args[1]));
//        System.exit(job.waitForCompletion(true) ? 0 : 1);
//    }
//}