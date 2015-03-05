#!/usr/bin/env python
import optparse
import sys,math
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=30, type="int", help="Limit on number of translations to consider per phrase (default=30)")
optparser.add_option("-s", "--stack-size", dest="s", default=9, type="int", help="Maximum stack size (default=9)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

threshold_limit = 5                 #for threshold pruning
alpha =0.9999                       #for reorder cost function
tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

def reorder(distance):              #return reordering cost
    return math.log(alpha**distance)
# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase,prev_end")
  #end of previous phrase added to hypothesis
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None,0)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  prev_end=0
  for i, stack in enumerate(stacks[:-1]):
    threshold = max(stack.itervalues(),key=lambda h: h.logprob).logprob*threshold_limit
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]:  # prune
      if(h.logprob>=threshold):                 #threshold and histogram - best of two
          n_end = len(f)-i if (len(f)-i)>0 else 1
          for n in xrange(0,n_end):#len(f)-i):
              for j in xrange(i+n+1,len(f)+1):
                  if(abs(i-j)>8):               #inside: first phrase chunk ends at j and second starts at i
                      continue
                  if f[i+n:j] in tm:
                      for phrase in tm[f[i+n:j]]:
                        logprob = h.logprob + phrase.logprob
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                          (lm_state, word_logprob) = lm.score(lm_state, word)
                          logprob += word_logprob
                        logprob += lm.end(lm_state) if j == len(f) else 0.0
                        logprob+= reorder(abs(i+n-h.prev_end))
                        new_hypothesis = hypothesis(logprob, lm_state, h, phrase,j)
                        prev_end = j
                        prev_logprob = logprob
                        prev_state = lm_state
                        if(n==0):               #save in stack if n=0 => no phrase skipped
                            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob:
                                    stacks[j][lm_state] = new_hypothesis
                                    prev_end = j
                        if f[i:i+n] in tm:              #compute skipped phrase
                            for phrase in tm[f[i:i+n]]:
                                logprob = prev_logprob+ phrase.logprob
                                lm_state = prev_state
                                for word in phrase.english.split():
                                    (lm_state, word_logprob) = lm.score(lm_state,word)
                                    logprob+= word_logprob
                                logprob+=lm.end(lm_state) if (i+n) == len(f) else 0.0
                                logprob+= reorder(abs(i-prev_end))
                                n_hypothesis = hypothesis(logprob,lm_state,new_hypothesis,phrase,i+n)
                                if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob:
                                    stacks[j][lm_state] = n_hypothesis

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  original= extract_english(winner)

  def extract_phrase_list(h):
      return [''] if h.predecessor is None else extract_phrase_list(h.predecessor) + [h.phrase.english]
  def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
  tm_logprob = extract_tm_logprob(winner)

  #reordering resultant sentence by one step look-ahead
  orig_phrase_list = extract_phrase_list(winner)
  new_state_list = [lm.begin()]
  total_new_score =0
  new_sentence  = [""]
  for index in xrange(1,len(orig_phrase_list)):
      context = new_state_list[index-1]
      look_ahead_context = new_state_list[index-1]
      curr = 0
      la_curr= 0
      for word in (orig_phrase_list[index].split()):
          (context,wlogprob)= lm.score(context,word)
          curr+=wlogprob
      if(index<len(orig_phrase_list)-1):
          for word in (orig_phrase_list[index+1].split()):
              (look_ahead_context,wlogprob)= lm.score(look_ahead_context,word)
              la_curr+=wlogprob
          if(la_curr>curr):
              new_sentence.append(orig_phrase_list[index+1])
              new_state_list.append(look_ahead_context)
              total_new_score+=la_curr
              total_new_score+= lm.end(look_ahead_context) if index == len(orig_phrase_list)-1 else 0.0
              orig_phrase_list[index+1] = orig_phrase_list[index]
              orig_phrase_list[index] = new_sentence[-1]
          else:
              new_sentence.append(orig_phrase_list[index])
              new_state_list.append(context)
              total_new_score+=curr
              total_new_score+= lm.end(context) if index == len(orig_phrase_list)-1 else 0.0
      else:
          new_sentence.append(orig_phrase_list[index])
          new_state_list.append(context)
          total_new_score+=curr
          total_new_score+= lm.end(context) if index == len(orig_phrase_list)-1 else 0.0

  #if reordering improves performance
  if(total_new_score>winner.logprob-tm_logprob):
      print ' '.join(new_sentence)
  else:
    print original
  if opts.verbose:
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
