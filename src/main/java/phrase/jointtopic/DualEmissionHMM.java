package phrase.jointtopic;

import edu.umass.nlp.functional.Fn;
import edu.umass.nlp.functional.Functional;
import edu.umass.nlp.io.IOUtils;
import edu.umass.nlp.ml.Regularizers;
import edu.umass.nlp.ml.sequence.*;
import edu.umass.nlp.optimize.LBFGSMinimizer;
import edu.umass.nlp.utils.*;
import org.apache.log4j.Logger;

import java.io.File;
import java.util.*;

public class DualEmissionHMM implements HMM {


  private transient Logger logger = Logger.getLogger("DualEmissionHMM");

  // H1, <s>, </s>

  private final Map<String, EmissionDistribution> wordEmissions;
  private final BasicEmissionDistribution background;
  private final CRF labelCRF;
  private final Map<String, TransitionDistribution> transitions;
  private final List<String> topicStates;
  private final List<Posteriors> posts;

  {              
    /* Topic States */
    topicStates = new ArrayList<String>();
    for (int k = 0; k < GlobalOptions.hmm.K; ++k) {
      String s = String.format("H%d", k);
      topicStates.add(s);
    }
//    /* Background Init */
//    background = new BasicEmissionDistribution("background", GlobalOptions.hmm.backgroundEmissionLambda);
//    for (Document document : Main.gDocuments) {
//      for (List<String> sent : document.getSentences()) {
//        background.observe(sent, 1.0);
//      }
//    }
  }



  private final StateSpace stateSpace;
  public static DualEmissionHMM global;

  public void write(String basePath) {
    this.write(basePath + File.separator + "crf",
               basePath + File.separator + "emissions",
               basePath + File.separator + "transitions");
  }

  protected  void write(String CRFfile, String emissionFile, String transitionFile) {
    IOUtils.writeObject(this.labelCRF, CRFfile);

    // write word emission counts
    List<String> emissions = new ArrayList<String>();
    for (IValued<String> wordPair : background.getEmissions()) {
      String word = wordPair.getFirst();
      // word itself
      StringBuilder sb = new StringBuilder(word).append(" ");
      // background distribution
      sb.append(wordPair.getSecond()).append(" ");
      //loop over other distributions
      for (String state : topicStates) {
        sb.append(wordEmissions.get(state).getEmissions().getCount(word)).append(" ");
      }
      emissions.add(sb.toString());
    }
    IOUtils.writeLines(emissionFile, emissions);

    List<String> transitions = new ArrayList<String>();
    for (String fromState : topicStates) {
      StringBuilder sb = new StringBuilder();
      for (String toState : topicStates) {
        sb.append(this.transitions.get(fromState).getTransitions().getCount(toState)).append(" ");
      }
      sb.append(this.transitions.get(fromState).getTransitions().getCount(StateSpace.stopLabel));
      transitions.add(sb.toString());
    }
    StringBuilder sb = new StringBuilder();
    for (String toState : topicStates) {
      sb.append(this.transitions.get(StateSpace.startLabel).getTransitions().getCount(toState)).append(" ");
    }
    sb.append(this.transitions.get(StateSpace.startLabel).getTransitions().getCount(StateSpace.stopLabel));
    transitions.add(sb.toString());
    IOUtils.writeLines(transitionFile, transitions);

  }

  public static DualEmissionHMM read(String basePath) {
    return DualEmissionHMM.read(basePath + File.separator + "crf",
                         basePath + File.separator + "emissions",
                         basePath + File.separator + "transitions");  
  }

  public static DualEmissionHMM read(String CRFfile, String emissionFile, String transitionFile) {
    DualEmissionHMM result = new DualEmissionHMM((CRF) IOUtils.readObject(CRFfile));
    List<String> emissions = IOUtils.lines(emissionFile);
    for (String emission : emissions) {
      String[] params = emission.split(" "); // params[0] is word, params[1] is background, params[2..n] are topics
      result.background.getEmissions().setCount(params[0], Double.parseDouble(params[1]));
      for (int i = 2; i < params.length; ++i) {
        // i-2 is adjustment for topic states starting at 0!
        result.wordEmissions.get(result.topicStates.get(i - 2)).getEmissions().setCount(params[0], Double.parseDouble(params[i]));
      }
    }
    // set global pointer -- this is hacky
//    for (String state : global.topicStates) {
//      ((SmoothedEmissionDistribution) global.wordEmissions.get(state)).global = global;
//    }

    List<String> transitions = IOUtils.lines(transitionFile);
    for (int i = 0; i < result.topicStates.size(); ++i) {
      String[] params = transitions.get(i).split(" ");
      for (int j = 0; j < result.topicStates.size(); ++j) {
        result.transitions.get(result.topicStates.get(i)).getTransitions().setCount(
          result.topicStates.get(j), Double.parseDouble(params[j]));
      }
      result.transitions.get(result.topicStates.get(i)).getTransitions().setCount(
        StateSpace.stopLabel, Double.parseDouble(params[params.length - 1]));
    }
    String[] params = transitions.get(result.topicStates.size()).split(" ");
    for (int j = 0; j < result.topicStates.size(); ++j) {
      result.transitions.get(StateSpace.startLabel).getTransitions().setCount(
        result.topicStates.get(j), Double.parseDouble(params[j]));
    }
    result.transitions.get(StateSpace.startLabel).getTransitions().setCount(
      StateSpace.stopLabel, Double.parseDouble(params[params.length - 1]));
    return result;
  }

  public String toString() {
    return this.toString(30);
  }

  public String toString(int x) {
    StringBuilder res = new StringBuilder("");
    for (String s : wordEmissions.keySet()) {
      res.append(s).append(": ").append(wordEmissions.get(s).toString(x)).append("\n");
    }
    res.append("Background: ").append(background.toString(x)).append("\n");
    res.append("Transitions:\n");
    for (Map.Entry<String, TransitionDistribution> entry : transitions.entrySet()) {
      res.append(entry.getKey() + ":" + entry.getValue().getTransitions()).append("\n");
    }
    return res.toString();
  }

  public Map<String, TransitionDistribution> getTransitions() {
    return transitions;
  }

  public Map<String, EmissionDistribution> getEmissions() {

    return wordEmissions;
  }

  public void observe(Posteriors posts) {
    assert posts.sentPosts.length == posts.document.getSentences().size();
    List<List<String>> sents = posts.document.getSentences();

    double scale = GlobalOptions.getDocumentScale(posts.document);

    int numSents = posts.sentPosts.length;

    for (int i = 1; i + 1 < numSents; i++) {
      for (int j = 0; j < GlobalOptions.hmm.K; j++) {
        this.wordEmissions.get(topicStates.get(j)).observe(sents.get(i), scale * posts.sentPosts[i][j]);
      }
    }

    for (Transition trans : stateSpace.getTransitions()) {
      this.transitions.get(trans.from.label).observe(trans.to.label,
        scale * posts.tranPosts.get(trans.from.label).getTransitions().getCount(trans.to.label));
    }

    if (posts.document.status == Document.Status.TRAIN_LABELED) {
      this.posts.add(posts);
    }


  }

  class LabeledDatum implements ILabeledSeqDatum {
      List<String> labels;
      List<String> sentence;
      List<List<String>> context;
      String topic;
      double weight;

      LabeledDatum(List<String> labels,
                   List<String> sentence,
                   String topic,
                   double weight) {
          this.labels = new ArrayList(labels);
          this.sentence = new ArrayList(sentence);
          this.topic = topic;
          this.weight = weight;
          this.context = null;
          assert (weight >= -1.e-04 && weight <= (1.0+1.0e-4));
      }

      LabeledDatum(List<String> labels,
                   List<String> sentence,
                   String topic,
                   double weight,
                   List<List<String>> context) {
        this(labels,sentence,topic,weight);
        this.context = context;
      }

      public List<String> getLabels() {
          if (Main.opts.crazy) {
            return Functional.map(labels, new Fn<String, String>() {
              public String apply(String input) {
                return input.endsWith("IMPORTANT") ?
                  topic :
                  input;

              }
            });
          }
          return labels;
      }

      public List<List<String>> getNodePredicates() {
        //TODO FIX ME
        List<List<String>> preds;
        if (context != null) preds = CRFData.getPredicates(sentence, context);
        else preds = CRFData.getPredicates(sentence);
        if (GlobalOptions.crf.useTopicFeatures)
            return CRFData.getTopicFeatures(preds, topic, global);
         else
            return preds;
      }

      public double getWeight() {
        return weight;
      }
  }


  public void makeDistribution() {
      if (GlobalOptions.isCRFTrainIter()) {


          List<ILabeledSeqDatum> data = new ArrayList<ILabeledSeqDatum>();

          for (Posteriors post : posts) {
              assert post.document.status == Document.Status.TRAIN_LABELED;
              List<List<String>> sents = post.document.getFullSents();
              List<List<String>> labels = post.document.getLabels();
              assert sents.size() == labels.size();

              for (int i = 0; i < sents.size(); ++i) {
                  switch (GlobalOptions.crf.trainTopicPosteriorMethod) {
                      case AVG_POSTERIORS:
                          for (int j = 0; j < topicStates.size(); ++j) {
                            if (Main.opts.crazy ||
                                GlobalOptions.crf.reuseWeights ||
                                post.sentPosts[i + 1][j] > GlobalOptions.crf.datumPostThresh)
                            {
                              List<List<String>> context = null;
                              if (GlobalOptions.crf.useAddlContextFeatures) context = post.document.getContext(i);
                              data.add(new LabeledDatum(labels.get(i), sents.get(i), topicStates.get(j),
                                  post.sentPosts[i + 1][j], context));
                            }
                          }
                          break;
                      case MAX_POSTERIORS:
                          int max = DoubleArrays.argMax(post.sentPosts[i+1]);
                          List<List<String>> context = null;
                          if (GlobalOptions.crf.useAddlContextFeatures) context = post.document.getContext(i);
                          data.add(new LabeledDatum(labels.get(i), sents.get(i), topicStates.get(max), 1, context));
                          break;
                      default:
                        throw new RuntimeException("Invalid topicPosteriorMethod!");
                  }
              }
          }

//          Iterable<ILabeledSeqDatum> data = Collections.toList(getCRFTrainData());

          logger.info("Start training CRF with " + data.size() + " instances");
          labelCRF.train(data,
            new CRF.Opts() {{
              this.regularizer = GlobalOptions.crf.hierRegularizer ? 
                  HierRegularizer.getRegularizer(labelCRF, GlobalOptions.crf.sigmaSquared,
                      GlobalOptions.crf.fineSigmaSquared) :
                  Regularizers.getL2Regularizer(GlobalOptions.crf.sigmaSquared);
              this.minPredCuttoffCount = 0;
              this.numThreads = GlobalOptions.iters.numThreads;
              this.cachePredTrainData = GlobalOptions.crf.cachePredTrainData;
              this.reuseWeights = GlobalOptions.crf.reuseWeights;
              this.optimizerOpts = new LBFGSMinimizer.Opts() {{
                this.minIters = GlobalOptions.iters.minCRFIters;
                this.maxIters = GlobalOptions.iters.maxCRFIters;
                this.maxHistoryResets = GlobalOptions.crf.maxHistoryResets;
                this.maxHistorySize = GlobalOptions.crf.maxHistorySize;
              }};
            }});
          logger.info("Done training CRF");
      }
  }

  public List<Posteriors> getPosts() {
    return posts;
  }

  public void addPosts(List<Posteriors> posts) {
    this.posts.addAll(posts);
  }

  public List<String> getTopicStates() {
    return topicStates;
  }

  public CRF getLabelCRF() {
    return labelCRF;
  }

  public EmissionDistribution getBackground() {
    return background;
  }

  public class Posteriors {

    // doc length
    final double[][] sentPosts;
    final Map<String, TransitionDistribution> tranPosts;
    final Document document;
    double logZ;

    Posteriors(Document doc) {
      this.sentPosts = new double[doc.getSentences().size()][GlobalOptions.hmm.K];
      this.tranPosts = getInitTransitions();
      this.logZ = Double.NEGATIVE_INFINITY;  // was 0.0
      this.document = doc;
    }
  }

  public Posteriors getRandomPosteriors(Document doc) {
    Posteriors dp = new Posteriors(doc);

    for (int i = 0; i < doc.getSentences().size(); i++) {
      for (int j = 0; j < GlobalOptions.hmm.K; j++) {
        dp.sentPosts[i][j] = GlobalOptions.rand.nextDouble();
      }
      DoubleArrays.probNormInPlace(dp.sentPosts[i]);
    }

    int numTrans = doc.getSentences().size() - 1;
    for (int i = 0; i < numTrans; ++i) {
      ICounter<Transition> cs = new MapCounter<Transition>();
      for (Transition trans : stateSpace.getTransitions()) {
        if (i == 0 && !trans.from.equals(stateSpace.startState)) continue;
        if (i > 0 && trans.from.equals(stateSpace.startState)) continue;
        if (i < numTrans - 1 && trans.to.equals(stateSpace.stopState)) continue;
        if (i == numTrans - 1 && trans.to.equals(stateSpace.stopState)) continue;
        cs.setCount(trans, GlobalOptions.rand.nextDouble());
      }
      cs = Counters.scale(cs, 1.0 / cs.totalCount());
      for (IValued<Transition> valued : cs) {
        Transition trans = valued.getElem();
        dp.tranPosts.get(trans.from.label).observe(trans.to.label, valued.getValue());
      }
    }

    return dp;
  }

  public Posteriors doInference(Document doc) {
      return doInference(doc, true);
  }

  public Posteriors doInference(Document doc, boolean useCRFParams) {
    // set up hmm problem, call forward backward
    double[][] scores = new double[doc.getSentences().size() - 1][stateSpace.getTransitions().size()];
    int numTrans = doc.getSentences().size() - 1;
    for (int i = 0; i < scores.length; ++i) {
      java.util.Arrays.fill(scores[i], Double.NEGATIVE_INFINITY);
      for (Transition trans : stateSpace.getTransitions()) {
        if (i == 0 && !trans.from.equals(stateSpace.startState)) continue;
        if (i > 0 && trans.from.equals(stateSpace.startState)) continue;
        if (i < numTrans - 1 && trans.to.equals(stateSpace.stopState)) continue;
        if (i == numTrans - 1 && !trans.to.equals(stateSpace.stopState)) continue;
        double obsLogProb = i > 0 ? wordEmissions.get(trans.from.label).getLogProbability(doc.getSentences().get(i))
          : 0.0;
        double transLogProb = transitions.get(trans.from.label).getLogProbability(trans.to.label);
        double CRFLogProb = 0.0;
        // Note the check is for iter greater than numStage1 so that
        // CRF has been trained at least once
        if (useCRFParams && GlobalOptions.isCRFReady() && i > 0 && doc.status == Document.Status.TRAIN_LABELED) {
          // i-1 because full sents don't have sentence begin/end padding
          List<List<String>> predicates;
          if (GlobalOptions.crf.useAddlContextFeatures) {
             predicates = CRFData.getPredicates(doc.getFullSents().get(i - 1), doc.getContext(i-1));
          }
          else {
            predicates = CRFData.getPredicates(doc.getFullSents().get(i - 1));
          }
          if (GlobalOptions.crf.useTopicFeatures) {
            predicates = CRFData.getTopicFeatures(predicates, trans.from.label, this.global);
          }
          CRFLogProb = labelCRF.getTagLogProb(
            Main.opts.crazy ?
            new LabeledDatum(doc.getLabels().get(i-1), doc.getFullSents().get(i-1), trans.from.label, 1.0) :
            new BasicLabelSeqDatum(predicates,doc.getLabels().get(i - 1), 1.0));

        }

        if (CRFLogProb < -1e-4 && GlobalOptions.hmm.ignoreLabeledParams) {
          scores[i][trans.index] = CRFLogProb;
        } else if (CRFLogProb < -1e-4) {                                                                                                          
          scores[i][trans.index] = GlobalOptions.hmm.unlabeledScale * (obsLogProb + transLogProb)
                                   + GlobalOptions.hmm.labeledScale * CRFLogProb;
        } else {
          scores[i][trans.index] = obsLogProb + transLogProb;
        }
      }
    }

    ForwardBackwards fb = new ForwardBackwards(stateSpace);
    ForwardBackwards.Result res = fb.compute(scores);

    Posteriors dp = new Posteriors(doc);

    for (int i = 0; i < doc.getSentences().size(); i++) {
      for (int j = 0; j < GlobalOptions.hmm.K; j++) {
        State s = stateSpace.getState(topicStates.get(j));
        dp.sentPosts[i][j] = res.stateMarginals[i][s.index];
      }
    }

    for (int i = 0; i < scores.length; ++i) {
      for (Transition trans : stateSpace.getTransitions()) {
        dp.tranPosts.get(trans.from.label).observe(trans.to.label, res.transMarginals[i][trans.index]);
      }
    }

    dp.logZ = res.logZ;

    return dp;
  }

  private Map<String, TransitionDistribution> getInitTransitions() {
    Map<String, TransitionDistribution> transitions = new HashMap<String, TransitionDistribution>();
    for (String topicState : topicStates) {
      transitions.put(topicState,
//        new StickyTransitionDistribution(topicState,
//                                         GlobalOptions.hmm.sigma,
//                                         GlobalOptions.hmm.transLambda));
        new BasicTransitionDistribution(GlobalOptions.hmm.transLambda));
    }
    transitions.put(StateSpace.startLabel,
//      new StickyTransitionDistribution(StateSpace.startLabel,
//                                       GlobalOptions.hmm.sigma,
//                                       GlobalOptions.hmm.transLambda));
      new BasicTransitionDistribution(GlobalOptions.hmm.startTransLambda));
    return transitions;
  }

  public DualEmissionHMM(CRF labelCRF) {
    this.labelCRF = labelCRF;
    this.background = new BasicEmissionDistribution("background", GlobalOptions.hmm.backgroundEmissionLambda);
    this.wordEmissions = new HashMap<String, EmissionDistribution>();
    for (String topicState : topicStates) {
      wordEmissions.put(topicState, new SmoothedEmissionDistribution(topicState, background, GlobalOptions.hmm.emissionLambda));
    }

    this.transitions = getInitTransitions();
    this.stateSpace = StateSpaces.makeFullStateSpace(topicStates);
    this.posts = new ArrayList<Posteriors>();
  }
}