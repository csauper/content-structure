Incorporating Content Structure into Text Analysis Applications
===============================================================

Christina Sauper      |  Aria Haghighi     | Regina Barzilay  
csauper@csail.mit.edu |  aria42@gmail.com  | regina@csail.mit.edu  

Abstract
--------

In this paper, we investigate how modeling content structure can benefit text
analysis applications such as extractive summarization and sentiment analysis.
This follows the linguistic intuition that rich contextual information should
be useful in these tasks. We present a framework which combines a supervised
text analysis application with the induction of latent content structure. Both
of these elements are learned jointly using the EM algorithm. The induced
content structure is learned from a large unannotated corpus and biased by the
underlying text analysis task. We demonstrate that exploiting content
structure yields significant improvements over approaches that rely only on
local context.

Full Text: http://groups.csail.mit.edu/rbg/code/content_structure/sauper-emnlp-10.pdf

Code
====

This code is available for research use only.

These instructions mainly pertain to the Amazon and Yelp data sets for
the multi-aspect phrase extraction task. 

Running
-------

Source code is included, but to just get started running the system quickly, I
recommend installing maven2, then compiling with `mvn compile` in the base
directory.  After that, run the system as follows (substituting your desired
config file and memory usage): 

java -ea -server -mx5G -Djava.ext.dirs=lib -cp target/classes phrase.jointtopic.Main amazon-conf.yaml


Config files
------------

amazon-conf.yaml
yelp-conf.yaml

  These are the config files which specify parameters for the model.  See
  GlobalOptions.java for more potential options.

conf/amazon.list
conf/yelp.list

  Lists of token files (see below for format).  Those marked as TRAIN_LABELED
  will be used as labeled input documents; those marked as TEST_LABELED will be
  used at test time.



Data
====

We performed experiments on three separate corpora, a set of Amazon HDTV
reviews (59 labeled, 12.8k unlabeled), a set of Yelp restaurant reviews (96
labeled, 31k unlabeled), and a set of IGN DVD reviews (665 labeled).

Formats
-------

*.tok

  Tokenized file, one word per line.  The columns of this file are as follows:
    word  sentence #  word #(sent)  start-char  end-char  start-char(sent)  end-char(sent)

*.ann
  
  Annotations corresponding to the tokenized file; one word's label per line.
  Some annotation files are tagged with begin / inside / end tokens; ability to
  automatically strip these is controlled by an option in the config file.
