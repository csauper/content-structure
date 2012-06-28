package phrase.jointtopic;

import edu.umass.nlp.exec.Execution;
import edu.umass.nlp.exec.Opt;
import edu.umass.nlp.functional.Fn;
import edu.umass.nlp.functional.Functional;
import edu.umass.nlp.functional.PredFn;
import edu.umass.nlp.io.IOUtils;
import edu.umass.nlp.io.ZipUtils;
import edu.umass.nlp.ml.F1Stats;
import edu.umass.nlp.ml.sequence.*;
import edu.umass.nlp.parallel.ParallelUtils;
import edu.umass.nlp.trees.ITree;
import edu.umass.nlp.trees.Trees;
import edu.umass.nlp.utils.*;
import edu.umass.nlp.utils.Collections;

import org.apache.log4j.FileAppender;
import org.apache.log4j.Logger;
import org.apache.log4j.SimpleLayout;

import java.io.File;
import java.util.*;
import java.util.zip.ZipFile;

public class Main {

  private static final Logger logger = Logger.getLogger("Main");

  public static List<Document> gDocuments;
  private static List<Document> gTestDocuments;
  private static List<Document> gTrainLabeledDocuments;
  public static Set<String> hmmVocab;

  public static Map<String, Integer> crfVocab;
  public static Map<String,String> pos;


  public static double doDocComputation(DualEmissionHMM suffStats, Document doc, boolean random) {
    DualEmissionHMM.Posteriors posts = random ?
      DualEmissionHMM.global.getRandomPosteriors(doc) :
      DualEmissionHMM.global.doInference(doc);
    suffStats.observe(posts);
    return posts.logZ;
  }

  public static double doDocComputations(DualEmissionHMM suffStats, Iterable<Document> docs, boolean random) {
    double logSum = 0.0;
    for (Document doc : docs) {
      doc.lastLogLike = GlobalOptions.getDocumentScale(doc) * doDocComputation(suffStats, doc, random);
      logSum += doc.lastLogLike;
    }
    return logSum;
  }


  private static double doIter() {

    reportUsedMemory("startIter");

    if (GlobalOptions.globalIter == GlobalOptions.iters.numStage1Iters) {
      logger.info("Starting Stage 2 Training: CRF Turned On");
    }

    double logLike;

    /* E-Step */
    DualEmissionHMM suffStats = new DualEmissionHMM(DualEmissionHMM.global.getLabelCRF());
    if (GlobalOptions.globalIter == 0 && GlobalOptions.hmm.randomize) {
      logLike = doDocComputations(suffStats, gDocuments, true);
    } else if (!GlobalOptions.hmm.parallelize) {
      // Side Effect: suffStats updated
      logLike = doDocComputations(suffStats, gDocuments, false);
    } else {
      logLike = doParallelIter(gDocuments, suffStats);
    }

    double unlabeledLogLike = 0.0;
    double labeledLogLike = 0.0;
    for (Document doc : gDocuments) {
      if (doc.status == Document.Status.TRAIN_LABELED) labeledLogLike += doc.lastLogLike;
      else unlabeledLogLike += doc.lastLogLike;
    }
    logger.info("Labeled Log Likelihood: " + labeledLogLike);
    logger.info("Unlabeled Log Likelihood: " + unlabeledLogLike);


    /* M-Step */
    reportUsedMemory("post-EStep");
    suffStats.makeDistribution();
    reportUsedMemory("post-makeDistribution");

    DualEmissionHMM.global = suffStats;

    /* Memory Clear Up */
    reportUsedMemory("post-GlobalReassign");
    suffStats.getPosts().clear();
    suffStats = null;
    reportUsedMemory("post-SuffStatsNulll");

    return logLike;
  }

  public static void learn() {


    double lastLike = Double.NEGATIVE_INFINITY;
    for (int iter = 0; iter < GlobalOptions.iters.numIters; ++iter) {

      /* EM */
      GlobalOptions.globalIter = iter;
      long startIterTime = System.currentTimeMillis();
      double logLike = doIter();
      long stopIterTime = System.currentTimeMillis();
      double iterSecs = (stopIterTime - startIterTime) / 1e3;


      /* Likelihood Check */
      if (GlobalOptions.globalIter > 1) {
        likelihoodCheck(lastLike, logLike);
        if (lastLike < logLike) lastLike = logLike;
      }      

      /* Report Stats */
      reportIterStats(iter, logLike, iterSecs);


      /* File Outputing */
      doTestAndCleanup(iter);
    }

  }

  private static void doTestAndCleanup(int iter) {

    if (canTestCRF()) {
      Logger trainEvalLogger = Logger.getLogger("Training Eval");
      trainEvalLogger.info("Training Eval");
      test(gTrainLabeledDocuments);
      trainEvalLogger.info("DONE");

      Logger testEvalLogger = Logger.getLogger("Testing Eval");
      test(gTestDocuments);
      testEvalLogger.info("DONE");
    }

    if (GlobalOptions.output.dir != null
        && (iter + 1 == GlobalOptions.iters.numIters || iter % GlobalOptions.output.printInterval == 0))
    {
      //reportUsedMemory("pre-OutputToFile");
      outputToFile();
      //reportUsedMemory("post-OutputToFile");
    }
  }

  private static boolean canTestCRF() {
    return DualEmissionHMM.global != null &&
      DualEmissionHMM.global.getLabelCRF() != null &&
      DualEmissionHMM.global.getLabelCRF().getLabels() != null;
  }

  private static void likelihoodCheck(double lastLike, double logLike) {
    // Make sure log likelihood isn't moving too far & is decreasing
    double diff = SloppyMath.relativeDifference(logLike, lastLike);
    logger.info(String.format("Log Likelihood difference: %.10f", diff));
    if (diff > 1.0e-4) {
      throw new RuntimeException(String.format("Bad likelihood: old: %.10f;  new: %.10f;  diff: %.10f",
        lastLike, logLike, diff));
    }

  }

  private static double doParallelIter(List<Document> docs, DualEmissionHMM suffStats) {
    double logLike = 0.0;
    class Worker implements Runnable {
      List<Document> docs;
      DualEmissionHMM suffStats;
      double logLike;


      public Worker(List<Document> docs) {
        this.docs = docs;

      }

      public void run() {
        suffStats = new DualEmissionHMM(DualEmissionHMM.global.getLabelCRF());
        if (docs.size() > 0) {
            logLike = doDocComputations(this.suffStats, docs, false);
        } else {
            logLike = 0;
        }
      }
    }

    List<List<Document>> parts = Collections.partition(docs, GlobalOptions.iters.numThreads);
    List<Worker> workers = Functional.map(parts, new Fn<List<Document>, Worker>() {
      public Worker apply(List<Document> input) {
        return new Worker(input);
      }
    });
    ParallelUtils.doParallelWork(workers, GlobalOptions.iters.numThreads);

    for (Worker worker : workers) {
      logLike += worker.logLike;
      for (String key : suffStats.getEmissions().keySet()) {
        Counters.incAll(suffStats.getEmissions().get(key).getEmissions(),
          worker.suffStats.getEmissions().get(key).getEmissions());
      }
      for (String key : suffStats.getTransitions().keySet()) {
        Counters.incAll(suffStats.getTransitions().get(key).getTransitions(),
          worker.suffStats.getTransitions().get(key).getTransitions());
      }
      Counters.incAll(suffStats.getBackground().getEmissions(), worker.suffStats.getBackground().getEmissions());
      suffStats.addPosts(worker.suffStats.getPosts());
    }
    return logLike;
  }

  private static void outputToFile() {
    logger.info("Serializing HMM");
    logger.info(GlobalOptions.output.dir.getPath());
    DualEmissionHMM.global.write(GlobalOptions.output.dir.getPath());
    DualEmissionHMM.global = null;
    System.gc();
    DualEmissionHMM.global = DualEmissionHMM.read(GlobalOptions.output.dir.getPath());
    logger.info("Finished");
  }

  private static void outputPosts(List<Document> docs, String label) {

    List<String> sentPosts = new ArrayList<String>();
    List<String> wordPosts = new ArrayList<String>();
    for (Document doc : docs) {
      DualEmissionHMM.Posteriors posts = DualEmissionHMM.global.doInference(doc, false);
      for (int i = 1; i < posts.sentPosts.length - 1; ++i) {
        String p = String.format("%d", i - 1);
        for (double post : posts.sentPosts[i]) {
          p += String.format(",%.4f", post);
        }
        sentPosts.add(p);
      }

      for (List<String> sent : doc.getSentences()) {
        for (String word : sent) {
          if (word.equals(StateSpace.startLabel) || word.equals(StateSpace.stopLabel)) continue;
          String p = String.format("\"%s\"", word.replace("\"", "\"\""));
          for (String topicState : DualEmissionHMM.global.getTopicStates()) {
            double post = DualEmissionHMM.global.getEmissions().get(topicState).getWordPosterior(
              word, DualEmissionHMM.global);
            p += String.format(",%.4f", post);
          }
          wordPosts.add(p);
        }
      }

      sentPosts.add("");
      wordPosts.add("");

    }

    logger.info("Writing posts to " + GlobalOptions.output.dir + File.separator + "{sent-posts,word_posts}." + label
      + "." + GlobalOptions.globalIter);
    IOUtils.writeLines(GlobalOptions.output.dir + File.separator + "sent_posts." + label, sentPosts);
    IOUtils.writeLines(GlobalOptions.output.dir + File.separator + "word_posts." + label, wordPosts);
  }

  private static void reportIterStats(int iter, double logLike, double iterSecs) {
    System.gc();
    double totalMemory = Runtime.getRuntime().totalMemory() / 1.0e6;
    double freeMemory = Runtime.getRuntime().freeMemory() / (1.0e6);
    logger.info("Iteration: " + iter);
    logger.info("Negative Log Likelihood: " + logLike);
    logger.info("Secs: " + iterSecs);
    logger.info("Free Megs: " + freeMemory);
    logger.info("Total Megs: " + totalMemory);
    logger.info("Used Megs: " + (totalMemory - freeMemory));
    logger.info("Current topics: \n" + DualEmissionHMM.global.toString());
  }

  private static void reportUsedMemory(String prefix) {
    System.gc();
    double totalMemory = Runtime.getRuntime().totalMemory() / 1.0e6;
    double freeMemory = Runtime.getRuntime().freeMemory() / (1.0e6);
//    logger.info(prefix + " Used: " + (totalMemory-freeMemory));
  }


  private static List<String> doMaxPosteriorsInferenceHelper(
    DualEmissionHMM dualHMM,
    List<List<String>> preds,
    double[] topicPosts,
    Map<String, edu.umass.nlp.utils.ICounter<String>> lossFn) {
    double[][] labelPosts = getCollapsedLabelPosts(dualHMM, preds, topicPosts);
    return DualEmissionHMM.global.getLabelCRF().getMinBayesRiskTagging(labelPosts, lossFn);
  }

  private static double[][] getCollapsedLabelPosts(DualEmissionHMM dualHMM, List<List<String>> preds, double[] topicPosts) {
    CRF crf = DualEmissionHMM.global.getLabelCRF();
    double[][] labelPosts = new double[preds.size()][crf.getLabels().size()];
    for (int k = 0; k < topicPosts.length; ++k) {
      double topicPost = topicPosts[k];
      if (topicPost == 0.0) continue;
      List<List<String>> topicPreds;
      if (GlobalOptions.crf.useTopicFeatures)
        topicPreds = CRFData.getTopicFeatures(preds, dualHMM.getTopicStates().get(k), dualHMM);
      else topicPreds = preds;
      ForwardBackwards.Result fbRes = crf.getResult(topicPreds);
      DoubleArrays.addInPlace(labelPosts, fbRes.stateMarginals, topicPost);
    }
    return labelPosts;
  }

//  private static Map<String, ICounter<String>> getLossFn() {
//    String path = GlobalOptions.inference.lossFnFile;
//    Map<String, ICounter<String>> res = new HashMap<String, ICounter<String>>();
//    for (String line : IOUtils.lines(path)) {
//      String[] fields = line.split("\\s+");
//      assert fields.length == 3;
//      String trueLabel = fields[0];
//      String guessLabel = fields[1];
//      double loss = Double.parseDouble(fields[2]);
//      Collections.getMut(res, trueLabel, new MapCounter<String>()).incCount(guessLabel, loss);
//    }
//    return res;
//  }

  private static List<String> stripBIETags(List<String> lst) {
    return Functional.map(lst, new Fn<String, String>() {
      public String apply(String input) {
        return input.replaceAll("[A-Z]-", "");
      }
    });
  }

  private static List<String> toBinary(List<String> lst, final String bg) {
    lst = Functional.filter(lst, new PredFn<String>() {
      public boolean holdsAt(String elem) {
        return !elem.equals("<S>") && !elem.equals("</S>");
      }
    });
    return Functional.map(lst, new Fn<String, String>() {
      public String apply(String input) {
        return input.equals(bg) ? bg : "BINARY";
      }
    });
  }

  private static List<String> getCrazyPredictions(List<List<String>> preds) {
    final CRF crf = DualEmissionHMM.global.getLabelCRF();
    ForwardBackwards.Result fbRes = crf.getResult(preds);
    List<String> res = new ArrayList<String>();
    res.add(StateSpace.startLabel);
    for (int i = 1; i + 1 < preds.size(); ++i) {
      ICounter<String> counter = new MapCounter<String>();
      final List<State> states = crf.getStateSpace().getStates();
      for (int s = 0; s < states.size(); ++s) {
        String state = states.get(s).label;
        if (state.startsWith("H")) {
          counter.incCount("IMPORTANT", fbRes.stateMarginals[i][s]);
        } else {
          counter.incCount(state, fbRes.stateMarginals[i][s]);
        }
      }
      if (counter.getCount("IMPORTANT") > 0.10) {
        res.add("IMPORTANT");
      } else {
        res.add("NONE");
      }
    }
    res.add(StateSpace.stopLabel);
    return res;
  }


  private static ICounter<IPair<Integer, Integer>> getTokenPosts(DualEmissionHMM.Posteriors posts) {
    Document doc = posts.document;
    ICounter<IPair<Integer, Integer>> res = new MapCounter<IPair<Integer, Integer>>();
    for (int i = 0; i < doc.getFullSents().size(); i++) {
      List<String> sent = doc.getFullSents().get(i);
      List<List<String>> preds;
      if (GlobalOptions.crf.useAddlContextFeatures) {
        preds = CRFData.getPredicates(sent, doc.getContext(i));
      }
      else preds = CRFData.getPredicates(sent);
      double[][] sentPosts = getCollapsedLabelPosts(DualEmissionHMM.global, preds, posts.sentPosts[i + 1]);
      for (int j = 1; j + 1 < sentPosts.length; ++j) {
        final List<State> states = DualEmissionHMM.global.getLabelCRF().getStateSpace().getStates();
        for (int s = 0; s < states.size(); ++s) {
          if (!states.get(s).label.equals("NONE")) {
            res.incCount(new BasicPair(i, j), sentPosts[j][s]);
          }
        }
      }
    }
    return res;
  }

  private static Set<IPair<Integer, Integer>> getLabeledPositions(Document doc) {
    Set<IPair<Integer, Integer>> res = new HashSet<IPair<Integer, Integer>>();
    for (int i = 0; i < doc.getFullSents().size(); i++) {
      List<String> labels = doc.getLabels().get(i);
      for (int j = 1; j + 1 < labels.size(); j++) {
        String label = labels.get(j);
        if (!label.equals("NONE")) {
          res.add(new BasicPair(i, j));
        }
      }
    }
    return res;
  }

  private static void testRougeF1(List<Document> docs) {
    double labeledTokCount = 0.0;
    double totalTokCount = 0.0;
    for (Document doc : docs) {
      labeledTokCount += doc.getNumLabels();
      totalTokCount += doc.getNumToks();
    }
    double avgLabToks = labeledTokCount / totalTokCount;
    Map<String, F1Stats> evalStats = new HashMap<String, F1Stats>();
    for (Document doc : docs) {
      List<List<String>> predictions = getBestPredictions(doc, (int) (doc.getNumToks() * avgLabToks));
      for (int i = 0; i < doc.getFullSents().size(); i++) {
        assert predictions.get(i).size() == doc.getLabels().get(i).size();
        TokenF1Eval.updateEval(evalStats, stripBIETags(doc.getLabels().get(i)), stripBIETags(predictions.get(i)));
        TokenF1Eval.updateEval(evalStats, toBinary(doc.getLabels().get(i), "NONE"), toBinary(predictions.get(i), "NONE"));
      }
    }
    logger.info("Avg Token Evaluation");
    for (Map.Entry<String, F1Stats> entry : evalStats.entrySet()) {
      final String label = entry.getKey();
      if (!Collections.set("NONE", StateSpace.startLabel, StateSpace.stopLabel).contains(label)) {
        logger.info(label + " -> " + entry.getValue().toString());
      }
    }
    logger.info("AVG -> " + TokenF1Eval.getAvgStats(evalStats, Collections.set("BINARY", "<S>", "</S>", "NONE")));
  }

  private static List<List<String>> getBestPredictions(Document doc, int numToks) {
    List<IValued<IPair<Integer, Integer>>> topK =
      Counters.getTopK(getTokenPosts(DualEmissionHMM.global.doInference(doc, false)), numToks);
    List<List<String>> predictions = new ArrayList<List<String>>();
    for (int i = 0; i < doc.getFullSents().size(); i++) {
      List<String> sent = doc.getFullSents().get(i);
      List<String> curPredictions = new ArrayList<String>();
      curPredictions.add(StateSpace.startLabel);
      for (int j = 0; j < sent.size(); j++) {
        curPredictions.add("NONE");
      }
      curPredictions.add(StateSpace.stopLabel);
      predictions.add(curPredictions);
    }
    for (IValued<IPair<Integer, Integer>> valued : topK) {
      int sentIndex = valued.getElem().getFirst();
      int tokIndex = valued.getElem().getSecond();
      assert tokIndex < predictions.get(sentIndex).size();
      predictions.get(sentIndex).set(tokIndex, "IMPORTANT");
    }
    return predictions;
  }

  private static void testRouge(List<Document> documents) {
    double correct = 0.0;
    double total = 0.0;
    Map<String, ICounter<String>> lossFn = getLossFn();
    for (int i = 0; i < documents.size(); ++i) {
      Document doc = documents.get(i);
      int numToks = doc.getNumLabels();
      Set<IPair<Integer, Integer>> correctPositions = getLabeledPositions(doc);
      List<IValued<IPair<Integer, Integer>>> topK =
        Counters.getTopK(getTokenPosts(DualEmissionHMM.global.doInference(doc, false)), numToks);
      for (IValued<IPair<Integer, Integer>> valued : topK) {
        if (correctPositions.contains(valued.getElem())) {
          correct += 1.0;
        }
        total += 1.0;
      }
    }
    logger.info(String.format("Rouge Recall: %.3f %d %d", correct / total, (int) correct, (int) total));
  }

  private static void test(List<Document> documents) {
//    logger.info("CRF Weights: " + java.util.Arrays.toString(DualEmissionHMM.global.getLabelCRF().getTransWeights()));
    testF1(documents);
    testRouge(documents);
//    try {
//      if (opts.binaryTask) testRougeF1(documents);
//    } catch (Exception e) {
//      logger.warn("Error with testRougeF1");
//      e.printStackTrace();
//    }
  }

  private static void testF1(List<Document> documents) {
    logger.info(GlobalOptions.output.dir.getPath());
    Map<String, ICounter<String>> lossFn = getLossFn();

    Map<String, F1Stats> evalStats = new HashMap<String, F1Stats>();
    List<String> debugOutput = new ArrayList<String>();
    for (int i = 0; i < documents.size(); ++i) {
      List<String> results = new ArrayList<String>();
      //List<String> data = new ArrayList<String>();
      Document doc = documents.get(i);
      List<List<String>> sents = doc.getFullSents();
      List<ITree<String>> trees = doc.getTrees();

      // get posteriors to find which is the best fit
      DualEmissionHMM.Posteriors posts = DualEmissionHMM.global.doInference(doc, false);
      for (int j = 1; j < posts.sentPosts.length - 1; ++j) {
        List<List<String>> preds;
        if (GlobalOptions.crf.useAddlContextFeatures)
          preds = CRFData.getPredicates(sents.get(j-1), doc.getContext(j-1));
        else
          preds = CRFData.getPredicates(sents.get(j - 1));
        List<String> predictions;
        List<Double> posteriors = null;
        double[] topicPosts = posts.sentPosts[j];
        assert (Math.abs(DoubleArrays.sum(topicPosts) - 1.0) < 1.0e-4);

        int max = DoubleArrays.argMax(posts.sentPosts[j]);
        final String maxTopic = DualEmissionHMM.global.getTopicStates().get(max);
        switch (GlobalOptions.crf.testTopicPosteriorMethod) {
          case MAX_POSTERIORS:
            java.util.Arrays.fill(topicPosts, 0.0);
            topicPosts[max] = 1.0;
            break;
          default:
            /* No Op */
        }
         List<String> outputPots = new ArrayList<String>();
        switch (GlobalOptions.inference.crfInferenceMode) {
          case VITERBI:
            assert GlobalOptions.crf.testTopicPosteriorMethod == GlobalOptions.TopicPosteriorMethod.MAX_POSTERIORS;
            List<List<String>> features = CRFData.getTopicFeatures(preds, maxTopic, DualEmissionHMM.global);
            //if (GlobalOptions.crf.useTopicFeatures) addToData(data, features);
            //else addToData(data, preds);
            predictions = DualEmissionHMM.global.getLabelCRF().getViterbiTagging(features);
            //outputPots = formatOutputPotentials(features);
            //pots = getOutputPotentials(features, predictions);
            break;
          case MIN_RISK:
            //if (GlobalOptions.crf.useTopicFeatures) addToData(data, features);
            //else
            //addToData(data, preds);
            predictions = doMaxPosteriorsInferenceHelper(DualEmissionHMM.global, preds, topicPosts, lossFn);
            posteriors = getOutputPosteriors(preds, predictions, topicPosts);
            break;
          case TREE:
            if (trees != null && Trees.getLeaves(trees.get(j-1)).size() == sents.get(j-1).size()) {
              assert j-1 < trees.size() : String.format("Invalid tree length at %s", sents.get(j - 1));
              Map<String,List<Double>> treePreds = getTreePredictions(trees.get(j-1), preds, topicPosts);
              for (String phrase : treePreds.keySet()) {
                StringBuilder sb = new StringBuilder(phrase);
                sb.append("\t");
                for (Double d : treePreds.get(phrase)) {
                  sb.append(d).append(" ");
                }
                sb.trimToSize();
                debugOutput.add(sb.toString());
              }
              debugOutput.add("");
              continue;
            }
            else {
              predictions = doMaxPosteriorsInferenceHelper(DualEmissionHMM.global, preds, topicPosts, lossFn);
              posteriors = getOutputPosteriors(preds, predictions, topicPosts);
            }
            break;
          case CRAZY:
            predictions = getCrazyPredictions(preds);
            //pots = getOutputPotentials(preds,predictions);
            break;
          default:
            throw new IllegalStateException();
        }
        for (int t = 0; t < doc.getFullSents().get(j - 1).size(); ++t) {
          String debugPots = outputPots.size() == doc.getFullSents().get(j-1).size() ?
              outputPots.get(t).replaceAll("\n"," %%% ") : "";
          debugOutput.add(String.format("%s %s %s %s %s %s", doc.getLabels().get(j - 1).get(t + 1), predictions.get(t + 1),
            doc.getFullSents().get(j - 1).get(t), maxTopic, posteriors != null ? posteriors.get(t+1) : "", debugPots));
        }
        debugOutput.add("");
//        results.addAll(predictions);
//        results.add("");
//        if (doc.isLabeled()) {
          TokenF1Eval.updateEval(evalStats, stripBIETags(doc.getLabels().get(j - 1)), stripBIETags(predictions));
          TokenF1Eval.updateEval(evalStats, toBinary(doc.getLabels().get(j - 1), "NONE"), toBinary(predictions, "NONE"));
//        }
      }


//      IOUtils.writeLines(GlobalOptions.output.dir + File.separator + "output" + i, results);
//      IOUtils.writeLines(GlobalOptions.output.dir + File.separator + "data" + i, data);
    }
    IOUtils.writeLines(Execution.getExecutionDirectory() + "/output", debugOutput);
    //IOUtils.writeLines(GlobalOptions.output.dir + File.separator + "output", debugOutput);
    logger.info("Wrote to: " + Execution.getExecutionDirectory() + "/output");

    if (opts.debugOutput) {
      IOUtils.writeLines(Execution.getExecutionDirectory() + "/topics." + GlobalOptions.globalIter,
          java.util.Collections.singletonList(DualEmissionHMM.global.toString(100)));
    }

    // output posteriors as well, if we're using topics
    if (opts.outputPosts && GlobalOptions.crf.useTopicFeatures)
      outputPosts(documents, "test");

    Logger evalLogger = Logger.getLogger("EVAL");
    try {
      evalLogger.addAppender(new FileAppender(new SimpleLayout(), opts.output + "/" + "eval"));
    } catch (Exception e) {
      e.printStackTrace();
    }

    logger.info("Evaluation");
    for (Map.Entry<String, F1Stats> entry : evalStats.entrySet()) {
      final String label = entry.getKey();
      if (!Collections.set("NONE", StateSpace.startLabel, StateSpace.stopLabel).contains(label)) {
        logger.info(label + " -> " + entry.getValue().toString());
      }
    }
    //logger.info("BINARY -> " + TokenF1Eval.getBinaryStats(evalStats));
    logger.info("AVG -> " + TokenF1Eval.getAvgStats(evalStats, Collections.set("BINARY", "<S>", "</S>", "NONE")));

  }

  private static Map<String,List<Double>> getTreePredictions(ITree root, List<List<String>> preds, double[] topicPosts) {
    double[][] labelPosts = getCollapsedLabelPosts(DualEmissionHMM.global, preds, topicPosts);
    List<String> labels = DualEmissionHMM.global.getLabelCRF().getLabels();
    List<String> goodLabels = Arrays.asList("S","SBAR","SQ","SBARQ","SINV","NP");

    Map<String,List<Double>> predictions = new HashMap<String,List<Double>>();
    Map<ITree<String>,Span> spans = Trees.getSpanMap(root);

    for (ITree<String> tree : spans.keySet()) {
      if (goodLabels.contains(tree.getLabel())) {

        StringBuilder sb = new StringBuilder();
        for (ITree<String> leaf : Trees.getLeaves(tree)) sb.append(leaf + " ");
        String label = sb.toString();

        List<Double> posts = new ArrayList<Double>();

        Span span = spans.get(tree);
        for (int i = span.getStart(); i < span.getStop(); ++i) {
          for (int j = 0; j < labels.size(); ++j) {
            if (i == span.getStart()) posts.add(Math.log(labelPosts[i+1][j]));
            else posts.set(j,posts.get(j)+Math.log(labelPosts[i+1][j]));
          }
        }

        predictions.put(label,posts);
      }
    }

    return predictions;
  }

  private static List<Double> getOutputPosteriors(List<List<String>> preds, List<String> predictions, double[] topicPosts) {
    List<Double> posteriors;
    posteriors = new ArrayList<Double>();
    double[][] labelPosts = getCollapsedLabelPosts(DualEmissionHMM.global, preds, topicPosts);
    List<String> labels = DualEmissionHMM.global.getLabelCRF().getLabels();
    for (int k = 0; k < predictions.size(); ++k) {
      posteriors.add(labelPosts[k][labels.indexOf(predictions.get(k))]);
    }

    return posteriors;
  }

  private static void addToData(List<String> data, List<List<String>> features) {
    for (List<String> feature : features) {
      StringBuilder sb = new StringBuilder();
      for (String pred : feature) {
        sb.append(pred).append(" ");
      }
      data.add(sb.toString());
    }
    data.add("");
  }


  private static Map<String, ICounter<String>> getLossFn() {
    //  assert GlobalOptions.inference.crfInferenceMode == GlobalOptions.CRFInferenceMode.MIN_RISK;
    logger.info("Building Loss Fn with recall penalty: " + GlobalOptions.inference.recallPenalty);
    Map<String, ICounter<String>> res = new HashMap<String, ICounter<String>>();
    for (String label : DualEmissionHMM.global.getLabelCRF().getLabels()) {
      for (String otherLabel : DualEmissionHMM.global.getLabelCRF().getLabels()) {
        String coarseLabel = label.replaceAll("^[A-Z]+-", "");
        String coarseOtherLabel = otherLabel.replaceAll("^[A-Z]+-", "");
        double loss;
        if (coarseLabel.equalsIgnoreCase(coarseOtherLabel)) {
          loss = 0.0;
        } else if (!label.equals("NONE") && otherLabel.equals("NONE")) {
          loss = GlobalOptions.inference.recallPenalty;
        } else {
          loss = 1.0;
        }
        Collections.getMut(res, label, new MapCounter<String>()).incCount(otherLabel, loss);
      }
    }
    return res;
  }

  public static class Opts {
    @Opt
    public String op;

    @Opt
    public String stopWords;

    @Opt
    public String pos;

    @Opt
    public String dataZip;

    @Opt
    public String fileList;

    @Opt
    public String output;

    @Opt
    public String devSet;

    @Opt
    public String inputPrefix;

    @Opt
    public int hmmDocWordCutoff = 1;

    @Opt
    public boolean binaryTask = false;

    @Opt
    public boolean stripBIE = false;

    @Opt
    public boolean crazy = false;

    @Opt
    public boolean outputPosts = true;

    @Opt
    public double hmmDocUpperBoundFrac = 1.0;

    @Opt
    public boolean debugOutput = false;
  }

  public static Opts opts;

  public static void main(String[] args) {
    /* Init Options */
    Execution.init(args[0]);
    opts = (Opts) Execution.fillOptions("main", new Opts());
    GlobalOptions.init();

    /* Crazy Consistency Check */
    if (opts.crazy) {
      assert !GlobalOptions.crf.useTopicFeatures;
      assert opts.binaryTask;
      assert opts.stripBIE;
      assert GlobalOptions.inference.crfInferenceMode == GlobalOptions.CRFInferenceMode.CRAZY;
    }

    /* Stop Words and Documents */
    crfVocab = new HashMap<String,Integer>();
    pos = new HashMap<String,String>();
    Set<String> stopWords = opts.stopWords != null ?
        new HashSet<String>(IOUtils.lines(opts.stopWords)) :
        new HashSet<String>();
    if (opts.pos != null) getPOS(IOUtils.lines(opts.pos));
        
    GlobalOptions.output.dir = new File(
      opts.output == null ?
        Execution.getExecutionDirectory() :
        opts.output);
    loadDocuments(opts.dataZip, IOUtils.lines(opts.fileList), stopWords);

    /* Load Model */
    if (opts.inputPrefix != null) {
      GlobalOptions.hmm.randomize = false;
      DualEmissionHMM.global = DualEmissionHMM.read(opts.inputPrefix);
    } else {
      DualEmissionHMM.global = new DualEmissionHMM(new CRF());
    }


    /* Execution Main */
    if (opts.op.equals("train")) {
      if (!GlobalOptions.output.dir.exists()) GlobalOptions.output.dir.mkdir();
      learn();
    } else if (opts.op.equals("test")) {
      test(gTestDocuments);
    } else {
      throw new RuntimeException("Invalid Operation: " + opts.op);
    }
  }

  private static void loadDocuments(String zipFilePath,
                                    List<String> fileListAndStatuses,
                                    Set<String> stopWords) {

    gDocuments = new ArrayList<Document>();
    gTrainLabeledDocuments = new ArrayList<Document>();
    gTestDocuments = new ArrayList<Document>();

    ICounter<String> dfCounts = new MapCounter<String>();

    ZipFile dataZip = ZipUtils.getZipFile(zipFilePath);
    for (String pathAndStatus : fileListAndStatuses) {
      String[] fields = pathAndStatus.split("\\s+");
      String tokFile = fields[0];
      if (!ZipUtils.entryExists(dataZip, tokFile)) {
        logger.warn("Couldn't find tok tokFile: " + tokFile);
        continue;
      }
      List<String> tokLines = IOUtils.lines(ZipUtils.getEntryInputStream(dataZip, tokFile));
      List<String> annotations = null;
      Document.Status status = Document.Status.UNLABELED;
      if (fields.length > 1 && fields[1].length() > 0) {
        try {
          status = Document.Status.valueOf(fields[1]);
        } catch (Exception e) {
          logger.debug("Not a valid status: \"" + fields[1] + "\"");
          System.exit(0);
        }
        String annfile = IOUtils.changeExt(tokFile, ".ann");
        if (!ZipUtils.entryExists(dataZip, annfile)) {
          throw new RuntimeException("No annotation found for " + status + " file " + tokFile);
        }
        annotations = getAnnotations(dataZip, annfile);
      }
      List<String> parses = null;
      String parsefile = IOUtils.changeExt(tokFile, ".parse");
      if (ZipUtils.entryExists(dataZip, parsefile)) {
        parses = IOUtils.lines(ZipUtils.getEntryInputStream(dataZip, parsefile));
      }
      final Document doc = new Document(tokFile.replaceAll(".tok", ""), tokLines, stopWords, annotations, parses, status);
      gDocuments.add(doc);
      switch (status) {
        case TRAIN_LABELED:
          gTrainLabeledDocuments.add(doc);
          break;
        case TEST_LABELED:
          gTestDocuments.add(doc);
          break;
      }
      Set<String> docWords = new HashSet<String>();
      for (List<String> sent : doc.getSentences()) {
        docWords.addAll(sent);
      }
      Counters.incAll(dfCounts, docWords);
    }
    Main.hmmVocab = Counters.getKeySet(dfCounts, new PredFn<IValued<String>>() {
      public boolean holdsAt(IValued<String> elem) {
        double frac = elem.getValue() / ((double) gDocuments.size());
        assert frac > 0.0 && frac <= 1.0;
        return elem.getValue() > opts.hmmDocWordCutoff &&
               frac <= opts.hmmDocUpperBoundFrac ;
      }
    });

    Logger logger = Logger.getLogger("Document Info");
    logger.info("Number of total documents: " + gDocuments.size());
    logger.info("Number of train labeled documents: " + gTrainLabeledDocuments.size());
    logger.info("Number of test labeled documents: " + gTestDocuments.size());
    logger.info("Word DF Count Cutoff: " + opts.hmmDocWordCutoff);
    logger.info("HMM Vocab Size: " + hmmVocab.size());

    /* Prune Document Sentences - relying
     * on the fact this changes refs for test docs too 
     *  */
    for (Document doc : gDocuments) {
      for (int i = 0; i < doc.getSentences().size(); i++) {
        List<String> oldSent = doc.getSentences().get(i);
        List<String> sent = Functional.filter(oldSent, new PredFn<String>() {
          public boolean holdsAt(String elem) {
            return hmmVocab.contains(elem);
          }
        });
        doc.setSentence(sent, i);
      }
    }
  }

  private static List<String> getAnnotations(ZipFile dataZip, String annfile) {
    List<String> annotations = IOUtils.lines(ZipUtils.getEntryInputStream(dataZip, annfile));
    if (annotations != null && opts.binaryTask) {
      annotations = Functional.map(annotations, new Fn<String, String>() {
        public String apply(String input) {
          return input.replaceAll("-.*$", "-IMPORTANT");
        }
      });
    }
    if (annotations != null && opts.stripBIE) {
      annotations = Functional.map(annotations, new Fn<String, String>() {
        public String apply(String input) {
          return input.replaceAll("[BIE]-", "");
        }
      });
    }
    return annotations;
  }

  private static void getPOS(List<String> poslist) {
    for (String line : poslist) {
      String[] strs = line.split(" ");
      Main.pos.put(strs[0],strs[1]);
    }
  }

}
