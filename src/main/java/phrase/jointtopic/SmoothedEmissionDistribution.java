package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

public class SmoothedEmissionDistribution implements EmissionDistribution, Serializable {
    private final BasicEmissionDistribution topicDistribution;
    private final BasicEmissionDistribution backgroundDistribution;
    private String topic;
    //public HMM global;

    public SmoothedEmissionDistribution(BasicEmissionDistribution topicDistribution,
                                        BasicEmissionDistribution backgroundDistribution) {
        this.topicDistribution = topicDistribution;
        this.backgroundDistribution = backgroundDistribution;
        this.topic = topicDistribution.getTopic();
        //this.global = global;
    }

    public SmoothedEmissionDistribution(String topic, BasicEmissionDistribution backgroundDistribution, double lambda) {

        this(new BasicEmissionDistribution(topic, lambda), backgroundDistribution);
    }

    public double getWordPosterior(String word, HMM global) {
        double tprob = ((SmoothedEmissionDistribution) global.getEmissions().get(topic)).topicDistribution.getEmissionProbability(word);
        double bprob = global.getBackground().getEmissionProbability(word);

        double a = (1- phrase.jointtopic.GlobalOptions.hmm.alpha) * tprob;
        double b = GlobalOptions.hmm.alpha * bprob;

        if (GlobalOptions.globalIter == 0 && word.equals("tv")) {
           boolean breakHere = true;
        }

        return a / (a + b);
    }

    public ICounter<String> getEmissions() {
        return topicDistribution.getEmissions();
    }

    public double getEmissionProbability(String emission) {
      assert Main.hmmVocab.contains(emission);
      return (1 - GlobalOptions.hmm.alpha) * topicDistribution.getEmissionProbability(emission) +
        GlobalOptions.hmm.alpha * backgroundDistribution.getEmissionProbability(emission);
    }

    public void observe(List<String> sent, double weight) {
        for (String word : sent) {
            double beta = getWordPosterior(word, DualEmissionHMM.global);            
            if (Double.isNaN(beta)) {
                assert GlobalOptions.globalIter == 0;
                topicDistribution.observe(Collections.singletonList(word), (1- GlobalOptions.hmm.alpha) * weight);
                backgroundDistribution.observe(Collections.singletonList(word), GlobalOptions.hmm.alpha * weight);
            }  else {
                topicDistribution.observe(Collections.singletonList(word), beta * weight);
                backgroundDistribution.observe(Collections.singletonList(word), (1-beta) * weight);

            }
        }
    }

    @Override
    public String toString() {
        return topicDistribution.toString();
    }

    public String toString(int x) {
        return topicDistribution.toString(x);
    }

    public double getLogProbability(List<String> sent) {

        double logProb = 0.0;

        for (String word : sent) {
            logProb += Math.log(getEmissionProbability(word));
        }

        assert logProb > Double.NEGATIVE_INFINITY;
        return logProb;
    }
}