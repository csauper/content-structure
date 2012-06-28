package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;
import edu.umass.nlp.utils.MapCounter;

import java.io.Serializable;

public class BasicTransitionDistribution implements TransitionDistribution, Serializable {
  private final ICounter<String> transitions;
  private final double lambda ;


  public BasicTransitionDistribution(ICounter<String> transitions, double lambda) {
    this.transitions = transitions;
    this.lambda = lambda;
  }


  public BasicTransitionDistribution(double lambda) {
    this(new MapCounter<String>(), lambda);
  }

  public void observe(String state, double weight) {
    transitions.incCount(state, weight);
  }

  public ICounter<String> getTransitions() {
    return transitions;
  }

  public double getLogProbability(String state) {

    double logProb = 0;

    double prob = (transitions.getCount(state) + lambda) /
        (transitions.totalCount() + lambda * (phrase.jointtopic.GlobalOptions.hmm.K+1.0));
    logProb += Math.log(prob);

    return logProb;
  }

  public double getTransitionProbability(String state) {
    return (transitions.getCount(state) + lambda) /
        (transitions.totalCount() + lambda * (phrase.jointtopic.GlobalOptions.hmm.K+1.0));
  }
}
