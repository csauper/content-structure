package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;

import java.util.List;

public interface EmissionDistribution {
    void observe(List<String> observations, double weight);
    double getLogProbability(List<String> strings);
    double getEmissionProbability(String emission);

    double getWordPosterior(String word, HMM global);
    ICounter<String> getEmissions();
    String toString(int x);
}
