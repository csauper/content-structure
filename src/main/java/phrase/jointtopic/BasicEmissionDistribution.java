package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;
import edu.umass.nlp.utils.MapCounter;

import java.io.Serializable;
import java.util.List;

public class BasicEmissionDistribution implements EmissionDistribution {
    private final ICounter<String> emissions;
    private final double lambda;

    public String getTopic() {
        return topic;
    }

    private final String topic;

    public BasicEmissionDistribution(String topic, ICounter<String> emissions, double lambda) {
        this.emissions = emissions;
        this.lambda = lambda;
        this.topic = topic;
    }

    public BasicEmissionDistribution(String topic, double lambda) {
        this(topic, new MapCounter<String>(), lambda);
    }

    public void observe(List<String> sent, double weight) {
        for (String word : sent ) {
            emissions.incCount(word, weight);
        }
    }

    public ICounter<String> getEmissions() {
        return emissions;
    }

    @Override
    public String toString() {
        return toString(30);
    }

    @Override
    public String toString(int x) {
      return emissions.toString(x);
    }

    public double getEmissionProbability(String word) {
      assert Main.hmmVocab.contains(word);
      return (emissions.getCount(word) + lambda) / (emissions.totalCount() + lambda * Main.hmmVocab.size());
    }

    public double getWordPosterior(String word, HMM ignored) {
        return getEmissionProbability(word);
    }

    public double getLogProbability(List<String> sent) {

        double logProb = 0;

        for (String word : sent) {
            double prob = getEmissionProbability(word);
            logProb += Math.log(prob);
        }

        return logProb;
    }
}
