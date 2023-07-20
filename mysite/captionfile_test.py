import webvtt
import re
import os
import math
import random

import collections
from collections import Counter
from random import randrange

def add_time(time, time_in_ms):
    # pre: string formatted time, int delay in ms
    # post: modified (added or subtract delay) time in string
    # hh:mm:ss.ttt
    ms = time_as_ms(time) + time_in_ms
    # now back to string format @svenMarnach from stackoverflow
    new_time = ms_as_time(ms)
    return new_time

def ms_as_time(ms):
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds = float(ms) / 1000
    new_time = "%i:%02i:%06.3f" % (hours, minutes, seconds)
    return new_time

def time_as_ms(time):
    time = time.replace(".", ":").split(":")
    hh, mm, ss, ttt = int(time[0]), int(time[1]), int(time[2]), int(time[3])
    ms = int(3600000 * hh + 60000 * mm + 1000 * ss) + ttt
    return ms

def get_wpm(caption):
    # word-per minute, total number of words by duration
    duration = time_as_ms(caption.end) - time_as_ms(
        caption.start
    )  # in miliseconds
    duration_in_min = duration / 60000  # 1 min = 60000 ms
    # number of words per time measure.
    wpm = len(caption.text.split()) / duration_in_min
    return wpm

def save_file(newfilename):
    # newfilename = str(filename.split(".")[0][:-2]) + ".vtt"
    vttfile.save(newfilename)
    with open(newfilename, "w") as fd:  # write to opened file
        vttfile.write(fd)

def set_delay(delay, caption_start=0):
    for cap_idx in range(len(vttfile)):
        if cap_idx >= caption_start:
            caption = vttfile[cap_idx]
            vttfile[cap_idx].start = add_time(caption.start, delay)
            vttfile[cap_idx].end = add_time(caption.end, delay)

def set_speed(new_wpm, caption_start=0):
    # set the speed of caption in word-per-minute manner
    # so that the average wpm to be the desired.
    # vtt = webvtt.read(filename) if isinstance(filename, str) else filename
    prev_end, prev_dt_ms = "", 0
    for cap_idx in range(len(vttfile)):
        caption = vttfile[cap_idx]
        wpm = get_wpm(caption)
        # calculate the current duration in the unit of minutes... 1 min = 60000 ms
        duration_in_min = math.fabs(
            (time_as_ms(caption.end) - time_as_ms(caption.start)) / 60000
        )

        if cap_idx >= caption_start:
            new_time = (
                len(caption.text.split()) / new_wpm
            )  # how much of time should it be changed to make the new wpm?
            dt_ms = (
                new_time - duration_in_min
            ) * 60000  # the delta time to be added/subtracted on
            # the new time mark.
            if prev_end:  # should be true after the first call...
                vttfile[cap_idx].start = prev_end
            vttfile[cap_idx].end = add_time(
                caption.end, dt_ms + prev_dt_ms
            )
            prev_dt_ms += dt_ms
            prev_end = vttfile[cap_idx].end


def fit_the_line(words):
    count = 0
    newcap = ""
    ## check if the words here fit the 32 character limit for each line
    print(words)
    for i,v in enumerate(words):
        print(i, v)
        count += len(v) + 1
        if count < 33:
            newcap += v + " "
        else:
            # add linebreak before this word
            print(f"this one:{i}")
            newcap += "\n" + v + " "
            count = 0
        print(f"count = {count}")
        print(newcap)

    return newcap


def set_mswords(rate, caption_start=0):
    # Remove RATE number of words from the caption-directly.
    missing_count = 0
    idx_list = list(range(caption_start, len(vttfile)))
    selected_idx = random.choice(idx_list)
    while missing_count < rate:
        tmp_words = vttfile[
            selected_idx
        ].text.split()
        # 1. clean out the linebreakers and
        #    get words from the caption-block
        words = [w.replace("\n", "") for w in tmp_words]

        leftwords = len(words)
        if leftwords > 2:
            pick_a_word = random.choice(words)
            words.remove(pick_a_word)            
            vttfile[selected_idx].text = fit_the_line(words) #" ".join(new_captxt)
            missing_count += 1
        else:  # when length of words initially < 2
            idx_list.remove(selected_idx)
            if not idx_list:  # if list is empty
                break  # Can't take out more words-
            else:
                selected_idx = random.choice(idx_list)  # pick another block,



def set_paraphrased(filename):
    # first, get the paraphrased file
    pf_fn = filename.replace("_0.vtt", "_100.vtt")
    vttfile = webvtt.read(pf_fn)    


vttfile = None  # initialization
iter_count = 0

filename = "caisapp/static/captions/base_captions/v1_sports_0.vtt"
value = [3000, 300, 1, 1, 2]


# Initializations
stop_count = 0
sentences = []
last_sentence = ""

# Now, initiate a caption file object
# First, pick the base file. (P = paraphrased, V = verbatim)
if value[3] == 1:
    # set_paraphrased(filename)
    vttfile = webvtt.read(filename.replace("_1.vtt", "_100.vtt"))    
else:
    vttfile = webvtt.read(filename)

# Pick last sentence from webvtt.read(filename), then apply set_delay, set_speed, set_mswords
# let's find how many English sentences in this caption
block_idx_with_stop = []
for cap_idx in range(len(vttfile)):
    caption = vttfile[cap_idx]
    stop_count += caption.text.count(".")
    if "." in caption.text:  # if current caption block is not the sentence-end,
        block_idx_with_stop.append(
            cap_idx
        )  # collect which caption block idx has the stop...

# print("block_idx_with_stop", block_idx_with_stop)
print("before")
for i in block_idx_with_stop:
    print(vttfile[i])

if stop_count <= 1:
    # if its only one sentence, apply speed and delay to whole vtt file
    # print("***** SINGLE SENTENCE VTT *****")
    target_sentence_start_idx = 0
else:
    # print("***** MULTI SENTENCE *****")
    # more than 1 sentence, that is, 2 or more sentences (or stops, i.e., periods)
    # check if second last sentence has stops in the middle
    if len(block_idx_with_stop) > 1:
        target_sentence_start_idx = block_idx_with_stop[-2] + 1

# print("target_sentence_start_idx:", target_sentence_start_idx)
set_delay(
    value[0], 0
)  # to apply delay from the beginning... as deb requested
set_speed(value[1], target_sentence_start_idx)
set_mswords(value[2], target_sentence_start_idx)

print("after")
for i in range(len(vttfile)):
    print(vttfile[i])



save_file("test.vtt")
