package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;

public interface TransitionDistribution {
    ICounter<String> getTransitions();
    void observe(String label, double count);
    double getLogProbability(String label);

    double getTransitionProbability(String s);
}
