package phrase.jointtopic;

import edu.umass.nlp.functional.Fn;
import edu.umass.nlp.ml.sequence.CRF;
import edu.umass.nlp.utils.BasicPair;
import edu.umass.nlp.utils.IPair;
import org.apache.commons.collections.primitives.ArrayIntList;
import org.apache.commons.collections.primitives.IntList;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class HierRegularizer {

  public static Fn<double[], IPair<Double, double[]>> getRegularizer(final CRF crf,
                                                                     final double coarseSigmaSq,
                                                                     final double fineSigmaSq) {
    return new Fn<double[], IPair<Double, double[]>>() {
      IntList[] coarseToFinePreds;
      //Map<Integer, List<Integer>> coarseToFinePreds;
      int numLabels;
      Pattern p = Pattern.compile(".*?_H\\d+$");

      private void initCoarsePreds() {
        coarseToFinePreds = new IntList[crf.getPredIndexer().size()];
        //new HashMap<Integer, List<Integer>>();
        for (int i = 0; i < crf.getPredIndexer().size(); ++i) {
          String pred = crf.getPredIndexer().get(i);
          Matcher m = p.matcher(pred);
          if (!m.matches()) {
            coarseToFinePreds[i] = new ArrayIntList();
            //coarseToFinePreds.put(i, new ArrayList<Integer>());
          }
        }
        for (int i = 0; i < crf.getPredIndexer().size(); ++i) {
          String pred = crf.getPredIndexer().get(i);
          Matcher m = p.matcher(pred);
          if (m.matches()) {
            String coarsePred = pred.replaceAll("_H\\d+$", "");
            int coarseIndex = crf.getPredIndexer().indexOf(coarsePred);
            assert coarseIndex >= 0;
            try {
              coarseToFinePreds[coarseIndex].add(i);
            } catch (Exception e) {
              System.err.println("No Coarse: " + pred);
            }
          }
        }
      }

      public IPair<Double, double[]> apply(double[] weights) {
        if (coarseToFinePreds == null) initCoarsePreds();
        double obj = 0.0;
        double[] grad = new double[weights.length];
        BitSet regd = new BitSet(weights.length);
        int numWeightsRegd = 0;
        for (int coarsePredIndex=0; coarsePredIndex < coarseToFinePreds.length; ++coarsePredIndex) {
          // not a coarse pred
          if (coarseToFinePreds[coarsePredIndex] == null) continue;
          //int coarsePredIndex = entry.getKey();
          IntList finePredIndices = coarseToFinePreds[coarsePredIndex];
          for (int label = 0; label < crf.getLabels().size(); ++label) {
            int coarseWeightIndex = crf.getNodeWeightIndex(coarsePredIndex, label);
            double coarseWeight = weights[coarseWeightIndex];
            obj += coarseWeight * coarseWeight / coarseSigmaSq;
            grad[coarseWeightIndex] += 2 * coarseWeight / coarseSigmaSq;
            regd.set(coarseWeightIndex);
            numWeightsRegd++;
            for (int i=0; i < finePredIndices.size(); ++i) {
              int finePredIndex = finePredIndices.get(i);
              int fineWeightIndex = crf.getNodeWeightIndex(finePredIndex, label);
              double fineWeight = weights[fineWeightIndex];
              double diff = fineWeight; 
                //(fineWeight - coarseWeight);
              obj += diff * diff / fineSigmaSq;
              grad[fineWeightIndex] += 2 * diff / fineSigmaSq;
               regd.set(fineWeightIndex);
              numWeightsRegd++;
            }
          }
        }
        for (int i=0; i < weights.length; ++i) {
          if (regd.get(i)) continue;
          double transWeight = weights[i];
          obj += transWeight * transWeight / coarseSigmaSq;
          grad[i] += 2 * transWeight / coarseSigmaSq;
          numWeightsRegd++;
        }
        assert numWeightsRegd == weights.length;
        return new BasicPair(obj, grad);
      }
    };
  }

}
