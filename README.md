content-structure
=================

Code from "Incorporating Content Structure into Text Analysis Applications"

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

Code
----
This code is available for research use only.

Data
----

We performed experiments on three separate corpora, a set of Amazon HDTV
reviews (59 labeled, 12.8k unlabeled), a set of Yelp restaurant reviews (96
labeled, 31k unlabeled), and a set of IGN DVD reviews (665 labeled).

