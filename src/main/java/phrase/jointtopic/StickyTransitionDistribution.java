package phrase.jointtopic;

import edu.umass.nlp.utils.ICounter;
import edu.umass.nlp.utils.MapCounter;


//TODO: implement this.
public class StickyTransitionDistribution implements TransitionDistribution {

  private final ICounter<String> newDistrCounts = new MapCounter<String>();
  private final String label;
  private final double sigma;
  private final double lambda;

  public StickyTransitionDistribution(String state, double sigma, double lambda) {
    this.lambda = lambda;
    this.sigma = sigma;
    this.label = state;
  }

  public void observe(String state, double weight) {
    if (!label.equals(this.label)) {
      newDistrCounts.incCount(label, weight);
    }
  }

  public ICounter<String> getTransitions() {
    return newDistrCounts;
  }

  // this is not right
  public double getTransitionProbability(String state) {
    if (state.equals(label)) {
      return sigma;
    } else {
      double newCount = newDistrCounts.getCount(state);
      double total = newDistrCounts.totalCount();
      double newProb = (newCount + lambda) / (phrase.jointtopic.GlobalOptions.hmm.K * lambda + total);
      return (1 - sigma) * newProb;
    }
  }

  // this is not right
  public double getLogProbability(String state) {
    return Math.log(getTransitionProbability(state));
//        double logProb = 0;
//
//        double prob = (transitions.getTransitions().getCount(state) + lambda) / (transitions.getTransitions().totalCount() + lambda * (GlobalOptions.K+1.0));
//        logProb += Math.log(prob);
//
//        return logProb;
  }
}