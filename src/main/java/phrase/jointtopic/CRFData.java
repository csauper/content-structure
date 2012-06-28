package phrase.jointtopic;

import edu.umass.nlp.ml.sequence.StateSpace;

import java.util.ArrayList;
import java.util.List;

public class CRFData {

  public static List<List<String>> getTopicFeatures(List<List<String>> predicates, String topic, HMM global) {
    List<List<String>> features = new ArrayList<List<String>>();


    int background = 0;
    int low = 0;
    int med = 0;
    int high = 0;
    int invalid = 0;

    for (List<String> predicate : predicates) {
      List<String> fs = new ArrayList<String>();      

      for (String s : predicate) {
        fs.add(new StringBuilder(s).append("_").append(topic).toString());
        fs.add(s);

        if (s.startsWith("WORD")) {
          StringBuilder sb = new StringBuilder("TOPIC_WORD_").append(topic);
          String[] fields = s.split("_");
          final String word = GlobalOptions.hmm.caseFold ? fields[1].toLowerCase() : fields[1];
          final String pos = fields[2];
          if (!Main.hmmVocab.contains(word)) {
            continue;
          }
          double post = global.getEmissions().get(topic).getWordPosterior(word,global);
//          if (GlobalOptions.rand.nextDouble() < 0.1) {
//            System.out.println(String.format("%s %s %s", topic, fields[1], post));
//          }
          if (post < 0.5) {
            fs.add(new StringBuilder("BACKGROUND_").append(pos).toString());
            background++;
          } else {
            fs.add(sb.append("_HIGH_").append(pos).toString());
            high++;
//            if (post < 0.66) {
//              fs.add(sb.append("_LOW_").append(pos).toString());
//              low++;
//            } else if (post < 0.83) {
//              fs.add(sb.append("_MED_").append(pos).toString());
//              med++;
//            } else if (post >= 0.83) {
//              fs.add(sb.append("_HIGH_").append(pos).toString());
//              high++;
//            } else {
//              invalid++;
//            }
          }
        }
      }
      features.add(fs);
    }

//    System.out.println(String.format("*%s* BG: %d  LOW: %d  MED: %d  HIGH: %d  INVALID: %d", topic, background, low, med, high, invalid));

    return features;
  }

  public static List<List<String>> getPredicates(List<String> sentence, List<List<String>> context) {
    List<List<String>> preds = getPredicates(sentence);

    // this is a total hack.  assuming we're passed in context where context[0] is prev sentence words,
    //    context[1] is this sent words, context[2] is next sent words.

    List<String> contextPreds = getAddlContextPreds(context);

    for (List<String> pred : preds) {
      pred.addAll(contextPreds);
    }

    return preds;
  }

  private static List<String> getAddlContextPreds(List<List<String>> context) {
    List<String> preds = new ArrayList<String>();

    for (String text : context.get(0)) {
      if ((Main.crfVocab.get(text) != null) && Main.crfVocab.get(text) >= GlobalOptions.crf.frequencyCutoff && text.matches(".*[a-zA-Z].*"))
        preds.add(new StringBuilder("PREV_CONTEXT_").append(text).toString());
    }
    for (String text : context.get(1)) {
      if ((Main.crfVocab.get(text) != null) && Main.crfVocab.get(text) >= GlobalOptions.crf.frequencyCutoff && text.matches(".*[a-zA-Z].*"))
        preds.add(new StringBuilder("THIS_CONTEXT_").append(text).toString());
    }
    for (String text : context.get(2)) {
      if ((Main.crfVocab.get(text) != null) && Main.crfVocab.get(text) >= GlobalOptions.crf.frequencyCutoff && text.matches(".*[a-zA-Z].*"))
        preds.add(new StringBuilder("NEXT_CONTEXT_").append(text).toString());
    }

    return preds;
  }

  public static List<List<String>> getPredicates(List<String> sentence) {
    List<List<String>> sentPreds = new ArrayList<List<String>>();
    List<String> preds = new ArrayList<String>();

    preds.add(StateSpace.startLabel);
    preds.addAll(getDirectionContext(sentence, -1, 1));
    sentPreds.add(preds);

    for (int i = 0; i < sentence.size(); ++i) {
      sentPreds.add(getContextFeatures(sentence,i));
    }

    preds = new ArrayList<String>();
    preds.add(StateSpace.stopLabel);
    preds.addAll(getDirectionContext(sentence,sentence.size(),-1));
    sentPreds.add(preds);

    return sentPreds;
  }



  private static List<String> getContextFeatures(List<String> text, int index) {
    List<String> preds = new ArrayList<String>();

    if (index >= 0 && index < text.size()) preds.addAll(getWordFeatures(text.get(index), 0));
    else if (index < 0) preds.add(StateSpace.startLabel);
    else preds.add(StateSpace.stopLabel);

    preds.addAll(getDirectionContext(text, index, -1));
    preds.addAll(getDirectionContext(text, index, 1));

    return preds;
  }

  private static List<String> getDirectionContext(List<String> text, int index, int direction) {
    List<String> preds = new ArrayList<String>();

    for (int i = 1; i <= GlobalOptions.crf.window; ++i) {
      int pos = index + (i * direction);
      if (pos >= 0 && pos < text.size()) preds.addAll(getWordFeatures(text.get(pos), i*direction));
      else if (pos < 0) preds.add(new StringBuilder(StateSpace.startLabel).append("_").append(i*direction).toString());
      else preds.add(new StringBuilder(StateSpace.stopLabel).append("_").append(i*direction).toString());
    }

    return preds;
  }

  private static List<String> getWordFeatures(String word, int i) {
    List<String> preds = new ArrayList<String>();
//    StringBuilder pos = (i == 0) ? new StringBuilder("") : new StringBuilder("_").append(i);
    StringBuilder pos = new StringBuilder("_").append(i);

    String origWord = word;

    if (GlobalOptions.crf.lowerCase) word = word.toLowerCase();
    // word itself
    preds.add(new StringBuilder("WORD_").append(word).append(pos).toString());

    if (Main.pos.get(word) != null) preds.add(new StringBuilder("POS_").append(Main.pos.get(word)).append(pos).toString());

    if (GlobalOptions.crf.lowerCaseFeature)
      preds.add(new StringBuilder("LC_").append(word).append(pos).toString());

    if (origWord.matches("^[A-Z]*$")) preds.add(new StringBuilder("ALL_CAPS").append(pos).toString());

    // word shape
    preds.add(new StringBuilder(word.replaceAll("[a-z]", "x").replaceAll("[A-Z]", "X")).append("_SHAPE").append(pos).toString());

    // initial cap
    if (origWord.matches("^[A-Z].*")) preds.add(new StringBuilder("INIT_CAP").append(pos).toString());

    // digit
    if (word.matches(".*\\d.*")) preds.add(new StringBuilder("HAS_DIGIT").append(pos).toString());
    // if (word.matches(".*\\..*")) preds.add(new StringBuilder("HAS_PERIOD").append(pos).toString());

    // suffix
    if (word.length() >= 3) preds.add(new StringBuilder(word.substring(word.length()-3)).append("_SUFFIX").append(pos).toString());
    else preds.add(new StringBuilder(word).append("_SUFFIX").append(pos).toString());

    preds.add(new StringBuilder("#BIAS").append(pos).toString());

    return preds;
  }

}
