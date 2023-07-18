import webvtt  # http://webvtt-py.readthedocs.io/en/latest/
import re
import os
import math
import random

import collections
from collections import Counter
from random import randrange
from django.conf import settings
from .models import Response, CaptionName


class CaptionFile:
    """Create a caption (.vtt) file with given
    Args:
        datafile (str): The file name with video genre (e.g., v4_sports_0.vtt)
        delay (int): delay in ms
        speed (int): speed in wpm
        mw (int): number of missing words
        pf (int): boolean value to represent whether the caption is verbatim (0) or paraphrased (1)
    Returns:
        exports a file
    """

    vttfile = None  # initialization
    iter_count = 0

    def __init__(self, filename, value):
        """
        Handle the file location to be stored -> static folder
        """
        genre = filename.rsplit("/", 1)[1].split("_")[1]
        version_number = filename.rsplit("/", 1)[1].split("_")[0]
        response = Response.objects.all().last()
        response_id = response.interview_uuid

        new_filename = str(
            settings.STATICFILES_DIRS[0]
        ) + "/captions/{}_{}_{}_{}.vtt".format(
            response_id, version_number, genre, self.iter_count
        )
        cap_title = "{}_{}_{}_{}".format(
            response_id, version_number, genre, self.iter_count
        )
        CaptionName.objects.create(response=response, caption_title=cap_title)

        # The iteration count 'must' match with the video clip quantity.
        if self.iter_count <= 20:
            self.iter_count += 1

        # Initializations
        stop_count = 0
        sentences = []
        last_sentence = ""

        # Now, initiate a caption file object
        # First, pick the base file. (P = paraphrased, V = verbatim)
        if value[3] == "p":
            self.set_paraphrased(filename)
        else:
            self.vttfile = webvtt.read(filename)

        # Pick last sentence from webvtt.read(filename), then apply set_delay, set_speed, set_mswords
        # let's find how many English sentences in this caption
        block_idx_with_stop = []
        for cap_idx in range(len(self.vttfile)):
            caption = self.vttfile[cap_idx]
            stop_count += caption.text.count(".")
            if "." in caption.text:  # if current caption block is not the sentence-end,
                block_idx_with_stop.append(
                    cap_idx
                )  # collect which caption block idx has the stop...

        # print("block_idx_with_stop", block_idx_with_stop)
        # for i in block_idx_with_stop:
        #     print(self.vttfile[i])

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
        self.set_delay(
            value[0], 0
        )  # to apply delay from the beginning... as deb requested
        self.set_speed(value[1], target_sentence_start_idx)
        self.set_mswords(value[2], target_sentence_start_idx)
        print("loading and rendering: {}".format(new_filename))
        # filename still ends up 0 cause paraphrasing was done internally (self.vttfile directly changed to the _100.vtt)
        self.save_file(new_filename)

    """############### HELPER FUNCTIONS ###############"""

    def save_file(self, newfilename):
        # newfilename = str(filename.split(".")[0][:-2]) + ".vtt"
        self.vttfile.save(newfilename)
        with open(newfilename, "w") as fd:  # write to opened file
            self.vttfile.write(fd)

    def add_time(self, time, time_in_ms):
        # pre: string formatted time, int delay in ms
        # post: modified (added or subtract delay) time in string
        # hh:mm:ss.ttt
        ms = self.time_as_ms(time) + time_in_ms
        # now back to string format @svenMarnach from stackoverflow
        new_time = self.ms_as_time(ms)
        return new_time

    def ms_as_time(self, ms):
        hours, ms = divmod(ms, 3600000)
        minutes, ms = divmod(ms, 60000)
        seconds = float(ms) / 1000
        new_time = "%i:%02i:%06.3f" % (hours, minutes, seconds)
        return new_time

    def time_as_ms(self, time):
        time = time.replace(".", ":").split(":")
        hh, mm, ss, ttt = int(time[0]), int(time[1]), int(time[2]), int(time[3])
        ms = int(3600000 * hh + 60000 * mm + 1000 * ss) + ttt
        return ms

    def get_wpm(self, caption):
        # word-per minute, total number of words by duration
        duration = self.time_as_ms(caption.end) - self.time_as_ms(
            caption.start
        )  # in miliseconds
        duration_in_min = duration / 60000  # 1 min = 60000 ms
        # number of words per time measure.
        wpm = len(caption.text.split()) / duration_in_min
        return wpm

    """####### main functions for variation #######"""

    def set_delay(self, delay, caption_start=0):
        for cap_idx in range(len(self.vttfile)):
            if cap_idx >= caption_start:
                caption = self.vttfile[cap_idx]
                self.vttfile[cap_idx].start = self.add_time(caption.start, delay)
                self.vttfile[cap_idx].end = self.add_time(caption.end, delay)

    def set_speed(self, new_wpm, caption_start=0):
        # set the speed of caption in word-per-minute manner
        # so that the average wpm to be the desired.
        # vtt = webvtt.read(filename) if isinstance(filename, str) else filename
        prev_end, prev_dt_ms = "", 0
        for cap_idx in range(len(self.vttfile)):
            caption = self.vttfile[cap_idx]
            wpm = self.get_wpm(caption)
            # calculate the current duration in the unit of minutes... 1 min = 60000 ms
            duration_in_min = math.fabs(
                (self.time_as_ms(caption.end) - self.time_as_ms(caption.start)) / 60000
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
                    self.vttfile[cap_idx].start = prev_end
                self.vttfile[cap_idx].end = self.add_time(
                    caption.end, dt_ms + prev_dt_ms
                )
                prev_dt_ms += dt_ms
                prev_end = self.vttfile[cap_idx].end

    def set_mswords(self, rate, caption_start=0):
        # remove RATE number of words from the caption-directly.
        missing_count = 0
        idx_list = list(range(caption_start, len(self.vttfile)))
        selected_idx = random.choice(idx_list)
        # selected_idx = random.randint(caption_start, len(self.vttfile) - 1) # pick a initial caption-block index
        while missing_count < rate:
            words = self.vttfile[
                selected_idx
            ].text.split()  # get words from the caption-block
            leftwords = len(words)

            if leftwords > 2:
                pick_a_word = random.choice(words)
                words.remove(pick_a_word)
                self.vttfile[selected_idx].text = " ".join(words)
                missing_count += 1

            else:  # when length of words initially < 2
                idx_list.remove(selected_idx)
                if not idx_list:  # if list is empty
                    break  # we cannot take any more words from the caption due to restriction we defined.
                else:
                    selected_idx = random.choice(idx_list)  # pick another block,

    def set_paraphrased(self, filename):
        # first, get the paraphrased file
        fn = filename.split("_")
        pf_fn = fn[0] + "_" + fn[1] + "_100.vtt"
        self.vttfile = webvtt.read(pf_fn)
