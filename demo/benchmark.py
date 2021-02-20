import webvtt 
import re
import string
import json
import itertools
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
from numpy import loadtxt
import pandas as pd
from sklearn.preprocessing import RobustScaler

from collections import Counter
from copy import deepcopy
from pathlib import Path

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.utils import plot_model

import modAL_multicol
from modAL_multicol.models import ActiveLearner, Committee
from modAL_multicol.multilabel import avg_score
from modAL_multicol.uncertainty import uncertainty_sampling

from mappings import cappy_ratings as cr
from mappings import contra_dicts as contractions

"""
    A = steno
    B = voice writer

    WC = word count
        Parameter	Weather	    NHL Hockey  SportsCentre    The Social
	        WPM	    191	        231	        269	            271
    Steno	CWPM	167	        155	        215	            164
            WC      493         378         528             377
	        CRTC	87.4%	    67.4%	    80.0%	        60.0%
	        NER	    98.9%	    98.2%	    98.9%	        99.6%
	        Delay	4.2 s	    4.0 s	    4.2 s	        3.3 s
					
    Voice	CWPM	167	        135	        193	            156
            WC      492         330         473             360
	        CRTC	87.2%	    58.7%	    71.8%	        57.6%
	        NER	    99.1%	    97.1%	    98.7%	        98.9%
	        Delay	5.6 s	    6.0 s   	6.0 s	        4.8 s

    weakness/assumptions made
    - transcript english sentence was assummed to be paraphrased/edited into 1-to-1 
    (i.e., each transcript sentences will have its caption counterpart)
    - English contractions made in captions were expanded to match with transcript...
"""

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def load_AL_models(f1, f2):
    n_members = 2 # initializing number of Committee members
    learner_list = list()

    orig_urls = [f1, f2]

    for member_idx in range(n_members):
        model_url = orig_urls[member_idx]
        print("\tLoaded Models from: {}\n".format(model_url))
        model = keras.models.load_model(model_url)  # load the classifier
        if member_idx == 0:
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # when model was loaded, we don't train extra x and y
        learner_list.append(ActiveLearner(estimator=model, query_strategy=avg_score))
    

    return Committee(learner_list=learner_list, given_classes=np.array([1,2,3,4,5]))

def to_ordinal(ratings):
    np_ratings = np.zeros(shape=(1, 20)) # in the shape of multiple columns, padd with zeros
    for c in [0,1,2,3]:
        tmp_start = c*5
        tmp_i = c*5 + ratings[c]
        for w in range(tmp_start, tmp_i):
            np_ratings[0, w] = 1
    return np_ratings

def get_sentences_manual(tr_vtt_file, cap_vtt_file):
    """
    Return: a list of lists where each element list has
        start block index, end block index, english sentence.
    
    The main problem solving task here is to segmentize English sentences,
    If a caption block has multiple english sentences (segmentized by either a question mark (?) or a period (.))
    then, we split the element and add the element separately to the output list...
    """
    
    output = []
    for v_file in [tr_vtt_file, cap_vtt_file]:
        idx_init = 0
        cap_chunks = []
        prev_sent = ""
        for cap_idx in range(0, len(v_file)):
            caption = v_file[cap_idx]
            # if current caption block is not the sentence-end,
            # We know the idx right after the end would be the beginning...    
            if ('.' in caption.text) or ('?' in caption.text):  
                tmp_sent = ' ' if prev_sent == '' else prev_sent
                for j in range(idx_init, cap_idx+1):
                    tp_txt = list(re.split('[\?|\.]', v_file[j].text))
                    sentence_counter = 0
                    if len(tp_txt) > 1:                    
                        while len(tp_txt[1:]) > 0:
                            if tmp_sent[-1] != ' ': # when sentence ends in 1 block
                                tmp_sent += ' ' + tp_txt[0].strip() + '. '
                            else:
                                tmp_sent += tp_txt[0].strip() + '. '
                            if sentence_counter > 0:
                                # if there are more then 1 sentence, we restart idx here.
                                idx_init = cap_idx
                            # print(idx_init, cap_idx, tmp_sent)

                            tmp_sent = tmp_sent.strip() # get rid of heading whitespace  
                            cap_chunks.append([idx_init, cap_idx, tmp_sent])
                            sentence_counter += 1
                            prev_sent = '. '.join(tp_txt[1:])
                            tmp_sent = ' '
                            tp_txt = list(re.split('[\?|\.]', prev_sent))
                    else: # continuing sentence
                        if tmp_sent[-1] != ' ':
                            tmp_sent += ' ' + tp_txt[0].strip()+' '
                        else:
                            tmp_sent += tp_txt[0].strip()+' '
                idx_init = cap_idx+1
        # for c in cap_chunks:
        #     print(c)
        output.append(cap_chunks)

    return output # tr_chunks, cap_chunks

def fps_to_ms(time):
    fps = round(int(time.split(":")[-1])/29.97*1000)
    if fps > 1000:
        fps = 0    
    if fps < 100: # pad with 0 if it's less than 100
        diff = 3-len(str(fps))
        padding = "0"*diff
        fps = padding+str(fps)
    tmp_st = [j for j in time.split(":")[:-1]]
    ntime = ":".join(tmp_st)
    ntime += "." + str(fps)    
    return ntime

def addtime(time):
    # add 1 second to the given time
    tmptime = time.split(":")
    if len(tmptime) > 3:
        ntime = tmptime[0]+":"+tmptime[1]+":"+str(int(tmptime[2])+1)+":"+tmptime[3]
    else: # possibly without the ms/frame unit
        ntime = tmptime[0]+":"+tmptime[1]+":"+str(int(tmptime[2])+1)
    return ntime

def sorted_words(sentence):
    tmp_l = []
    for s in sentence.split(" "):
        tmp_l.append(s.translate(str.maketrans('', '', string.punctuation)))    
    return sorted(tmp_l)

def to_ms(captiontime):
    tmpct = captiontime.split(':')
    tmpms = tmpct[2].split('.')
    hour, minutes, second, milliseconds = int(tmpct[0]), int(tmpct[1]), int(tmpms[0]), int(tmpms[1])
    tmp_t = 0
    if milliseconds > 0:
        tmp_t += milliseconds
    tmp_t += (hour * 60 * 60 * 1000) + (minutes * 60 * 1000) + (second * 1000)
    return tmp_t

def get_wpm(startcaption, endcaption, cap_txt):
    # word-per minute, total number of words by duration
    duration = to_ms(endcaption.end)-to_ms(startcaption.start)  # in miliseconds
    duration_in_min = duration/60000  # 1 min = 60000 ms
    # number of words per time measure.
    wpm = len(cap_txt.split())/duration_in_min
    
    # print(duration, duration_in_min, len(cap_txt.split()))
    return wpm

def get_block_avgwpm(startcaption_idx, endcaption_idx, cap_vtt):
    # for each caption block, starting from start idx to end idx,
    # calculate the wpm for the block, then
    # get average value to be returned.
    wpm = 0
    for i in range(startcaption_idx, endcaption_idx+1):
        this_cap = cap_vtt[i]
        tmp_wpm = get_wpm(this_cap, this_cap, this_cap.text)

        # print(this_cap.start, this_cap.end, this_cap.text, tmp_wpm)
        wpm += tmp_wpm
    
    wpm = wpm / (endcaption_idx-startcaption_idx+1)

    return wpm

def find_error_words(tr_txt, cap_txt=[], print_switch=False):
    """
        compare transcript and caption,
        return missing words and error words...
    """
    missing_words, error_words = 0, 0

    if cap_txt: # default mode.
        matches = [] # let's try spliting words in cap_txt and words in tr_txt
        sorted_tr_txt, sorted_cap_txt = sorted_words(tr_txt), sorted_words(cap_txt)
        """below is simple version"""
        for w in sorted_cap_txt:
            if w in sorted_tr_txt:
                sorted_tr_txt.remove(w)
                matches.append(w)
        missing_words += len(sorted_tr_txt) # things left in transcript

        for w1 in matches:
            sorted_cap_txt.remove(w1)
        
        if print_switch:
            print("Transcript:", tr_txt)
            print("Caption:", cap_txt)
            print('Matched words:', matches)
            print('Word Errors:', sorted_cap_txt)
            print("Words not captioned:", sorted_tr_txt)
            print("\tMissing {} Words".format(missing_words))
            variable = input('Press Enter to see next: ')
        
    else: # we compare only one,
        for word in sorted_words(tr_txt):
            missing_words += 0 if word == '' else 1

    return missing_words


def benchmark_prep(tr_vtt, cap_vtt):
    """"
        To compare two files, we first find sentences and its start and ending block indices.
        1. we match the sentences. This is important because we must find the indices from one to another.
        ## number of missing 'captions', because there can be multiple captions paraphrased into one or two.
        ## or can be cases where a whole 'english sentence' is not existing.
    """
    rep_values = []
    tr_chunks, cap_chunks = get_sentences_manual(tr_vtt, cap_vtt)
    # diff = len(tr_chunks) - len(cap_chunks) 
    # print('The DIFF={}'.format(diff))    
    i,c,stuck_c = 0,0,0
    skipped = False
    while i < len(tr_chunks):
        delay, speed, missing_words, paraphrasing = 0,0,0,0
        cb, use_sim = "", 0
        if c < len(cap_chunks):
            tr_block, cap_block = tr_chunks[i], cap_chunks[c]
            tr_txt, cap_txt = tr_block[2], cap_block[2]
            # first, let's handle anything within the brackets [] or ()
            tr_txt, cap_txt = tr_txt.lower(), cap_txt.lower()
            if "]" in tr_txt:
                tr_txt = " ".join(tr_txt.split("]")[1:])
            if "]" in cap_txt:
                cap_txt = " ".join(cap_txt.split("]")[1:])
            # next, handle english contractions
            for coin in contractions:
                if coin in tr_txt:
                    tr_txt = decontracted(tr_txt)
                if coin in cap_txt:
                    cap_txt = decontracted(cap_txt)

            transcript_sentence, caption_sentence = nlp(tr_txt), nlp(cap_txt) # lets compare the two sentences by using NLP
            use_sim = transcript_sentence.similarity(caption_sentence) # use the similarity method that is based on the vectors, on Doc, Span or Token
            delay = to_ms(cap_vtt[cap_block[0]].start) - to_ms(tr_vtt[tr_block[0]].start) # actual caption start time - transcript start time
            speed = get_block_avgwpm(cap_block[0], cap_block[1], cap_vtt)
            # speed = get_wpm(cap_vtt[cap_block[0]], cap_vtt[cap_block[1]], cap_block[2])
            if use_sim < 0: # lets' consider the sentence is actually not correct at all
                # print("the caption was probably skipped.") # then, set values to 0 & move i next.
                paraphrasing = 1 # it's edited, after all...
                skipped = True
            elif use_sim < 1: # if it's paraphrased, the value is 1 otherwise 0
                paraphrasing = 1 
                missing_words = find_error_words(tr_txt, cap_txt) # Find the number of missing words
                skipped = False
            else:
                skipped = False
            cb = cap_block[2]
        else: # this is when there are more transcript but not captioned...            
            tr_block, cap_block = tr_chunks[i], cap_chunks[c]
            tr_txt, cap_txt = tr_block[2], cap_block[2]
            delay = -1
            speed = 0            
            missing_words = find_error_words(tr_txt) # Find the number of missing words
            paraphrasing = 1
        
        tr_msg = "[{} -> {}]: {}".format(tr_vtt[tr_block[0]].start, tr_vtt[tr_block[1]].end, tr_block[2])
        cap_msg = "[{} -> {}]: {}".format(cap_vtt[cap_block[0]].start, cap_vtt[cap_block[1]].end, cap_block[2])
        
        rep_values.append([tr_msg, cap_msg, use_sim, delay, speed, missing_words, paraphrasing])
        # Assume that the captions can have less blocks/sentences..
        if i < len(tr_chunks):
            i += 1            
            if c < len(cap_chunks)-1:
                c = i
            else:
                c = len(cap_chunks)-1 # case when there are no more caption blocks
    return rep_values

def get_cappy_score(repvalues, scaler_x, flag=False):
    hearing_group = {"Deaf": 1, "Hard of Hearing": 2} # for a group mark
    cappy_score_rack = {"Deaf": [], "Hard of Hearing": []}
    factors = ["delay", "speed", "missing words", "caption paraphrasing"]
    
    for r in rep_values:
        tr_b, cap_b, use_sim, delay, speed, mw, pf = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        pf_txt = "Paraphrased" if pf == 1 else "Verbatim"
        # output_msg = '{tblock}\n{capblock}\n\n USE similarity = {usesim:0.3f}%\n D = {d} ms, S = {s:3.0f} WPM, Missing {m} words, {p}.'.format(
        #     tblock=tr_b, capblock=cap_b, usesim=use_sim*100, d=delay, s=speed, m=mw, p=pf_txt)
        # print(output_msg)
        output_msg = '{tblock}\n{capblock}\n\n D = {d} ms, S = {s:3.0f} WPM, Missing {m} words, {p}.'.format(
            tblock=tr_b, capblock=cap_b, d=delay, s=speed, m=mw, p=pf_txt)
        print(output_msg)
        ratemsg = ""
        for hearing_condition in hearing_group:
            if delay > 0:
                val = scaler_x.transform(np.asarray([[delay, speed, mw, pf, hearing_group[hearing_condition]]]))
                op = [x+1 for x in cappy.predict(val)]
            else: # if caption was not even displayed, it's automatic worst quality
                op = [1, 1, 1, 1]
            cappy_score_rack[hearing_condition].append(op)
            ratemsg += "\n\t{} viewers might be:\n\t\t".format(hearing_condition)
            ratingsentence = ""
            for idx in range(len(op)):
                caprates = op[idx]
                # ratingsentence += "{} with the {} \n\t\t".format(cr[caprates], factors[idx])
                ratingsentence += "{}:{} with {}\n".format(caprates, cr[caprates], factors[idx])
            ratemsg += ratingsentence
            # if hearing_condition == "Deaf":
            #     print(ratingsentence)
            # if op[1] == 2 and hearing_condition == "Hard of Hearing":
            # if flag:
            #     print(output_msg)
            #     print(ratemsg)
        output_msg += '\nCappy predicted that: ' + ratemsg
        
        # print()

        # print(output_msg)
    ## now, lets get the overall (average score)    
    for r in cappy_score_rack:
        this_group = cappy_score_rack[r]
        ratemsg = ""
        d,s,m,p = 0,0,0,0
        for ratings in this_group:
            d += ratings[0]
            s += ratings[1]
            m += ratings[2]
            p += ratings[3]
        avgd, avgs, avgm, avgp = d/len(this_group), s/len(this_group), m/len(this_group), p/len(this_group)
        avgop = [avgd, avgs, avgm, avgp]
        
        
        for idx in range(len(op)):
            caprates = avgop[idx]
            ratemsg += "- {:0.3f}/5, {} with the {}\n".format(caprates, cr[round(caprates)], factors[idx])
            # ratemsg += "- {:0.3f} with the {}\n".format(caprates, factors[idx])
            # ratemsg += "- {} with the {} and {}\n".format(cr[round(caprates)], factors[idx], caprates)
        print("Overall ratings by {} viewers might be:\n{}\nAverage can be:{}".format(r, ratemsg, np.mean(avgop)))

    print("TOTAL NUMBER OF ENGLISH SENTENCES:", len(rep_values))
    t_d, t_s, t_mw = 0,0,0
    for r in rep_values:
        t_d += r[3]
        t_s += r[4]
        t_mw += r[5]
    print("AVERAGE delay={:0.3f}, speed={:0.3f}, missing words/sentences={:0.3f}".format(t_d/len(rep_values), t_s/len(rep_values), t_mw/len(rep_values)))


if __name__ == "__main__":
    caption_files = [
        "captions/GlobalWeather_A.vtt", "captions/GlobalWeather_B.vtt",
        "captions/NHL_A.vtt", "captions/NHL_B.vtt",
        "captions/SportsCentre_A.vtt", "captions/SportsCentre_B.vtt",
        "captions/TheSocial_A.vtt", "captions/TheSocial_B.vtt",
    ]

    transcript_files = [
        "transcripts/GlobalWeather_tc.vtt", "transcripts/NHL_tc.vtt",
        "transcripts/SportsCentre_tc.vtt", "transcripts/TheSocial_tc.vtt"
    ]


    # ## LOAD language models for the uniserversal encoder
    import spacy_universal_sentence_encoder
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    # ## LOAD Cappy
    cappy = load_AL_models("data_and_models/originalfirst.h5", "data_and_models/originalsecond.h5")    



    deaf_csv_url = "data_and_models/deaf_20000_used.csv"
    tmp_dataset = np.array(pd.read_csv(deaf_csv_url, header=None, sep=',')) # actually need exception because it's i/o
    scaler_x = RobustScaler()
    cpy_xpool = scaler_x.fit_transform(tmp_dataset[:, :5]) 

    # Task 3. Now, we compare the caption file to transcript file.
    for ver in (['A', 'B']):
        for t in transcript_files:
            tr_dir, version = t, ver
            cap_dir = 'captions/' + tr_dir.split('/')[1].split('_')[0] + '_' + version + '.vtt'        
            tr_vtt, cap_vtt = webvtt.read(tr_dir), webvtt.read(cap_dir)
            print(cap_dir)
            rep_values = benchmark_prep(tr_vtt, cap_vtt) # handle benchmark prep
            cappy_values = get_cappy_score(rep_values, scaler_x, flag=True)

    # t = "transcripts/TheSocial_tc.vtt"
    # tr_dir, version = t, "B"
    # cap_dir = 'captions/' + tr_dir.split('/')[1].split('_')[0] + '_' + version + '.vtt'        
    # tr_vtt, cap_vtt = webvtt.read(tr_dir), webvtt.read(cap_dir)
    # print(cap_dir)
    # rep_values = benchmark_prep(tr_vtt, cap_vtt) # handle benchmark prep
    # cappy_values = get_cappy_score(rep_values, scaler_x, flag=True)
