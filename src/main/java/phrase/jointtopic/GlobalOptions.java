package phrase.jointtopic;

import edu.umass.nlp.exec.Execution;
import edu.umass.nlp.exec.Opt;

import java.io.File;
import java.util.Random;

public class GlobalOptions {

  public static class IterOpts {
    @Opt
    public int numIters = 101;

    @Opt
    public int numStage1Iters = 20; // HMM iters before starting CRF

    @Opt
    public int CRFInterval = 5; // train after every N iterations of HMM

    @Opt
    public int minCRFIters = 5;

    @Opt
    public int maxCRFIters = 150;

    @Opt
    public double annScale = 1.0;

    @Opt
    public int numThreads = Runtime.getRuntime().availableProcessors();
  }

  public static IterOpts iters = new IterOpts();

  public static class HMMOpts  {
    @Opt
    public int K = 5; // topics

    @Opt
    public boolean randomize = true; // randomize posteriors? false for continuing

    @Opt
    public int randSeed = 0XDEADBEEF;

    @Opt
    public double emissionLambda = 1.0;  // smoothing param

    @Opt
    public double backgroundEmissionLambda = 1.0;  // smoothing param

    @Opt
    public double startTransLambda = 1.0;

    @Opt
    public double transLambda = 1.0;
        
    @Opt
    public boolean parallelize = true;

    @Opt
    public double alpha = 0.5;

    @Opt
    public double labeledScale = 1;
    
    @Opt
    public boolean ignoreLabeledParams = false;

    @Opt
    public boolean caseFold = true;

    @Opt
    public boolean ignoreStops = true;
   
    @Opt
    public double unlabeledScale = 1.0;
  }

  public static HMMOpts hmm = new HMMOpts();

  public static class CRFOpts {

    @Opt
    public int window = 3;  // window for crf params

    @Opt
    public boolean useTopicFeatures = true;

    @Opt
    public double sigmaSquared = 1.0;

    @Opt
    public boolean reuseWeights = false;

    @Opt
    public TopicPosteriorMethod trainTopicPosteriorMethod = TopicPosteriorMethod.AVG_POSTERIORS;

    @Opt
    public TopicPosteriorMethod testTopicPosteriorMethod = TopicPosteriorMethod.AVG_POSTERIORS;

    @Opt
    public boolean cachePredTrainData = true;

    @Opt
    public int maxHistoryResets = 0;

    @Opt
    public boolean lowerCase = true;

    @Opt
    public double datumPostThresh = 1.0e-4;

    @Opt
    public int maxHistorySize = 3;

    @Opt
    public boolean lowerCaseFeature = false;

    @Opt
    public boolean hierRegularizer = false;

    @Opt
    public double fineSigmaSquared = 0.1;

    @Opt
    public boolean useAddlContextFeatures = false;

    @Opt
    public int frequencyCutoff = 5;
  }

  public static CRFOpts crf = new CRFOpts();

  public static class OutputOpts {
    @Opt
    public File dir = null;
    @Opt
    public int printInterval = 10;
  }

  public static OutputOpts output = new OutputOpts();

  public static enum TopicPosteriorMethod {
    MAX_POSTERIORS, AVG_POSTERIORS
  }

  public static enum CRFInferenceMode {
    VITERBI, MIN_RISK, TREE, CRAZY
  }

  public static class InferenceOpts {
    @Opt
    public CRFInferenceMode crfInferenceMode = CRFInferenceMode.MIN_RISK;

    @Opt
    public double recallPenalty = 5.0;
  }

  public static InferenceOpts inference = new InferenceOpts();

  // NOT AN OPTION
  public static int globalIter = -1;
  public static Random rand ;

  public static double getDocumentScale(Document doc) {
    return isCRFReady() && doc.status == Document.Status.TRAIN_LABELED ? GlobalOptions.iters.annScale : 1.0;
  }

  public static boolean isCRFReady() {
    int stage2Iter = GlobalOptions.globalIter - GlobalOptions.iters.numStage1Iters;
    return (stage2Iter > 0);
  }

  public static boolean isCRFTrainIter() {
    // TODO: should probably check for HMM initialized (op == continue || iter > 0).
    int stage2Iter = GlobalOptions.globalIter - GlobalOptions.iters.numStage1Iters;
    return (stage2Iter >= 0 && stage2Iter % GlobalOptions.iters.CRFInterval == 0);
  }

  public static void init() {
    Execution.fillOptions("iters", iters);
    Execution.fillOptions("hmm", hmm);
    Execution.fillOptions("crf", crf);
    Execution.fillOptions("output", output);
    Execution.fillOptions("inference", inference);
    rand = new Random(hmm.randSeed);
  }




}
