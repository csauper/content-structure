package phrase.jointtopic;

import edu.umass.nlp.functional.Fn;
import edu.umass.nlp.functional.Functional;
import edu.umass.nlp.ml.sequence.StateSpace;
import edu.umass.nlp.trees.ITree;
import edu.umass.nlp.trees.Trees;
import phrase.jointtopic.CRFData;

import java.util.*;

public class Document {


  public static enum Status {
    UNLABELED, TRAIN_LABELED, TEST_LABELED
  }

  private List<List<String>> sentences;
  private List<List<String>> labels;
  private List<List<String>> fullSents;
  private List<ITree<String>> trees;
  public double lastLogLike;
  public final Status status;
  public final String id;

  public boolean isLabeled() {
    return status != Status.UNLABELED;
  }



  public List<List<String>> getFullSents() {
    assert status != Status.UNLABELED;
    return fullSents;
  }

  public List<ITree<String>> getTrees() {
    return trees;
  }


  //Creates a document from a tokenized file
  public Document(String id,
                  List<String> lines,
                  Set<String> stopWords,
                  List<String> annotations,
                  List<String> parses,
                  Status status)
  {
    if (status != Status.UNLABELED && annotations == null) {
      throw new RuntimeException("No annotation for " + status + " document ");
    }
    this.status = status;
    this.id = id;

    assert annotations == null || annotations.size() == lines.size();

    this.sentences = new ArrayList<List<String>>();
    this.fullSents =
        annotations != null ?
            new ArrayList<List<String>>() :
            null;
    this.trees =
        parses != null && isLabeled() ?
            new ArrayList<ITree<String>>() :
            null;
    //this.predicates = new ArrayList<List<List<String>>>();
    if (annotations != null) this.labels = new ArrayList<List<String>>();

    List<String> start = new ArrayList<String>();
    start.add(StateSpace.startLabel);
    sentences.add(start);

    List<String> sent = new ArrayList<String>();
    List<String> fullSent = new ArrayList<String>();
    List<String> label = new ArrayList<String>();
    String sentNum = "0";
    if (annotations != null) label.add(StateSpace.startLabel);


    for (int i = 0; i < lines.size(); i++) {
      String[] params = lines.get(i).split(" ");
      if (!params[1].equals(sentNum)) {
        this.sentences.add(sent);


        assert parses == null || Integer.parseInt(sentNum) < parses.size() : String.format("Parse tree error on sentence: %s", sent);
        if (parses != null && isLabeled()) trees.add(Trees.readTree(parses.get(Integer.parseInt(sentNum))));
        if (fullSents != null) this.fullSents.add(fullSent);
        //this.predicates.add(CRFData.getPredicates(fullSent));
        if (annotations != null) {
          label.add(StateSpace.stopLabel);
          this.labels.add(label);
          label = new ArrayList<String>();
          label.add(StateSpace.startLabel);
        }
        sent = new ArrayList<String>();
        fullSent = new ArrayList<String>();
      }
      String word = params[0];
      // lowercase, remove stop words, require >= 1 alphabetic char
      if ((!GlobalOptions.hmm.ignoreStops || !stopWords.contains(word.toLowerCase())) && word.matches(".*[a-zA-Z].*")) {
        sent.add(GlobalOptions.hmm.caseFold ? word.toLowerCase() : word);
      }
      fullSent.add(word);
      if (!stopWords.contains(word.toLowerCase()) && Main.crfVocab.get(word) != null)
        Main.crfVocab.put(word, Main.crfVocab.get(word) + 1);
      else if (!stopWords.contains(word.toLowerCase())) Main.crfVocab.put(word,1);
      if (annotations != null) label.add(annotations.get(i));

      sentNum = params[1];
    }
    this.sentences.add(sent);
    if (parses != null && isLabeled()) trees.add(Trees.readTree(parses.get(Integer.parseInt(sentNum))));
    if (fullSents != null) this.fullSents.add(fullSent);
    //this.predicates.add(CRFData.getPredicates(fullSent));
    if (annotations != null) {
      label.add(StateSpace.stopLabel);
      this.labels.add(label);
    }

    List<String> stop = new ArrayList<String>();
    stop.add(StateSpace.stopLabel);
    sentences.add(stop);

    assert sentences.size() > 2;
//        assert annotations == null || labels.size() > 2;
  }

//    public Document(List<String> lines, Set<String> stopWords) {
//        this(lines, stopWords, null);
//    }
//
//    public Document(List<String> lines) {
//        this(lines, new HashSet<String>());
//    }

  public List<List<String>> getSentences() {
    return sentences;
  }

  public void setSentence(List<String> sent, int i) {
    sentences.set(i, sent);
  }

  public List<List<String>> getLabels() {
    assert status != Status.UNLABELED;
    return labels;
  }

//    public List<List<List<String>>> getPreds() {
//        return predicates;
//    }

  public int getNumLabels() {
    int count = 0;
    for (List<String> labelSent : labels) {
      for (int i=1; i+1 < labelSent.size(); ++i) {
        String label = labelSent.get(i);
        if (!label.equals("NONE")) {
          count++;
        }
      }
    }
    return count;
  }

  public int getNumToks() {
    int count = 0;
    for (List<String> labelSent : labels) {
      count += labelSent.size();
    }
    return count;
  }

  public List<List<String>> getContext(int i) {
    List<List<String>> context = new ArrayList<List<String>>(3);

    if (i-1 >= 0) context.add(fullSents.get(i-1));
    else context.add(new ArrayList<String>());

    context.add(fullSents.get(i));

    if (i + 1 < fullSents.size()) context.add(fullSents.get(i+1));
    else context.add(new ArrayList<String>());

    return context;
  }
}
