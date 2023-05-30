from django.http import HttpResponse, HttpResponseRedirect 
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ResponseForm
from multiprocessing import Pool
from random import randrange
from django.core import serializers
from .models import (
    Question,
    AnswerVideo,
    Response,
    Category,
    VideoGenerator,
    Blobby,
    CaptionName,
)
from django.apps import apps
from django.core.serializers.json import DjangoJSONEncoder
from django.forms.models import model_to_dict

import datetime
import time
import socket, errno


ORIGINALH5_ONE = settings.BASE_DIR + "/originalfirst.h5"
ORIGINALH5_TWO = settings.BASE_DIR + "/originalsecond.h5"
MODIFIEDH5_ONE = settings.BASE_DIR + "/modifiedfirst.h5"
MODIFIEDH5_TWO = settings.BASE_DIR + "/modifiedsecond.h5"
import random
import json
import filecmp
import numpy as np
from datetime import date
from .caption_file import CaptionFile


VIDCOUNT = 0
WAIT_SWITCH = False
CUR_PREDS = 0
CUR_QINSTANCE = 0 
VIDEO_TITLE = ""
CAPTION_TITLE = ""
LIST_NUMBERS = []
WATCHED_VIDEOS = []
VIDEO_POOL = []
VIDEO_CATEGORY = ""
UUID = ""
TRIVIA_QA = ""
MANUAL_IDX = []
MANUAL_COUNTER = 1

def consent(request):
    if request.method == "POST" or None:
        return redirect(survey)

    return render(request, "consent.html")

def survey(request):
    form = ResponseForm(request.POST or None)
    context = {"form": form}
    global UUID, LIST_NUMBERS, MANUAL_IDX, VIDEO_POOL, VIDCOUNT, VIDEO_CATEGORY, WAIT_SWITCH
    UUID = form.uuid
    if form.is_valid():
        form.save()
        LIST_NUMBERS = list(range(0, 81))
        MANUAL_IDX = sorted(random.sample(range(2, 17), 2)) # pick two numbers to insert the manual one in between..
        VIDEO_POOL = json.loads(serializers.serialize('json', VideoGenerator.objects.all()))
        VIDCOUNT = len(VIDEO_POOL)
        if Category.objects.filter(name="Video").count() < 1:
            Category.objects.create(name="Video")
        VIDEO_CATEGORY = Category.objects.filter(name="Video")[0].name
        WAIT_SWITCH = True
        return redirect(index)
    else:
        print(form.errors)
    return render(request, "survey.html", context)


def index(request):
    max_vid_count = 20 # 20 videos
    global MANUAL_IDX, VIDEO_POOL, LIST_NUMBERS, TRIVIA_QA, CUR_PREDS, CUR_QINSTANCE, VIDEO_TITLE, WATCHED_VIDEOS, CAPTION_TITLE
    
    if len(VIDEO_POOL) == 0: # when video pool is empty
        return redirect(byebye)        
    else:          
        if WAIT_SWITCH:
<<<<<<< HEAD
            print("WAIT SWITCH@index", WAIT_SWITCH)
            if apps.get_app_config("home").count == 0:
                apps.get_app_config("home").set_x_pool() # initiate the dataset..?
            # 1. get prediction from cappy backend
            q_instance, preds, queried_vals = apps.get_app_config("home").make_prediction()
            
            print(q_instance, preds, queried_vals.astype(int))
            global CUR_PREDS
            CUR_PREDS = preds
            global CUR_QINSTANCE
            CUR_QINSTANCE = q_instance

            if q_instance is None: # when there's no query_idx provided, we close the case
                return render(request, "byebye.html")
            # 2. parse the prediction passed from cappy.
            preds = list(map(lambda x: int(x), preds))
            preds = json.dumps(preds)
            # 3. load the video
            rand_video_sess = random.choice(VIDEO_POOL)
            global VIDEO_TITLE
            VIDEO_TITLE = rand_video_sess['fields']['video_name']
            VIDEO_POOL.remove(rand_video_sess)
            global WATCHED_VIDEOS
            WATCHED_VIDEOS.append(rand_video_sess)
            
            print("VIDEO_POOL-length:{}, collection: {}".format(len(VIDEO_POOL), [vd['fields']['video_name'] for vd in WATCHED_VIDEOS]))
            #  pick the two numbers for the trivia questions..
            global LIST_NUMBERS
            rand_numbs = random.sample(LIST_NUMBERS, 2)
            global TRIVIA_QA
=======
            rand_numbs = random.sample(LIST_NUMBERS, 2) # Pick the two numbers for the trivia questions..
>>>>>>> b1183d68109cdd4107b40e1a47fecc9feb88cdf8
            TRIVIA_QA = (rand_numbs)
            LIST_NUMBERS.remove(rand_numbs[0])
            LIST_NUMBERS.remove(rand_numbs[1])
            if apps.get_app_config("home").count == 0:
                apps.get_app_config("home").set_x_pool() # initiate the dataset..?
            
            if apps.get_app_config("home").count in MANUAL_IDX:
                # test1 has a poor caption, and the predicted ratings will be [5,5,5,5]
                # test2 has a good caption (transcript), but the predicted ratings will be [1,1,1,1]
                tmpidx = MANUAL_COUNTER
                preds = '[1,1,1,1]' if tmpidx == 2 else '[5,5,5,5]'
                CUR_PREDS = preds
                VIDEO_TITLE, CAPTION_TITLE = "/videos/test{}.mp4".format(tmpidx), "/captions/base_captions/test{}.vtt".format(tmpidx)
                print("Learn Count={} MANUAL_IDX={} MANUAL_COUNTER={} preds={}".format(apps.get_app_config("home").count, MANUAL_IDX, MANUAL_COUNTER, preds))

            else:
                # 1. get prediction from cappy backend
                q_instance, preds, queried_vals = apps.get_app_config("home").make_prediction()
                print(q_instance, preds, queried_vals.astype(int))
                CUR_PREDS, CUR_QINSTANCE = preds, q_instance
                # 2. parse the prediction passed from cappy.
                preds = list(map(lambda x: int(x), preds))
                preds = json.dumps(preds)
                # 3. load the video
                rand_video_sess = random.choice(VIDEO_POOL)
                VIDEO_TITLE = rand_video_sess['fields']['video_name']
                VIDEO_POOL.remove(rand_video_sess)
                WATCHED_VIDEOS.append(rand_video_sess)
                print("VIDEO_POOL-length:{}, collection: {}".format(len(VIDEO_POOL), [vd['fields']['video_name'] for vd in WATCHED_VIDEOS]))
                # 4. get url to pass the CaptionFile obj
                url = settings.STATICFILES_DIRS[0] + "/captions/base_captions/{}".format(VIDEO_TITLE.split("/")[1].split(".")[0] + "_0.vtt")
                CaptionFile(url, queried_vals)
                CAPTION_TITLE = "captions/{}.vtt".format(CaptionName.objects.last().caption_title)
            
            # 5. passing context values to initiate rendering        
            context = {
                "triviaQuestions": TRIVIA_QA, "submitReady": apps.get_app_config("home").count,
                "vid_count": 22, "preds": preds, "videourl": VIDEO_TITLE, "caption_title": CAPTION_TITLE,
            }
            return render(request, "index.html", context)
        else:
            time.sleep(5)
            return index(request)


def client_to_view(request):
    if request.method == "POST":
        print("now in client_to_view function@views.py")
        global WAIT_SWITCH
        WAIT_SWITCH = False
        print("WAIT SWITCH@client_to_view={} manidx={}".format(WAIT_SWITCH, MANUAL_IDX))
        client_rating = json.loads(request.POST["client_id"])
        client_rating = [int(i) for i in client_rating]
        client_list = list(map(lambda x: int(x), client_rating))
        category = Category.objects.filter(name=VIDEO_CATEGORY)[0]
        q = Question.objects.filter(text="videoresp")[0]
        resp = Response.objects.filter(interview_uuid=UUID).last()
        
        if apps.get_app_config("home").count in MANUAL_IDX: 
            print("Learn Count={} MANUAL_IDX={}".format(apps.get_app_config("home").count, MANUAL_IDX))
            queried_val = [[0,0,0,0]]
            apps.get_app_config("home").count = apps.get_app_config("home").count + 1
            global MANUAL_COUNTER
            MANUAL_COUNTER = MANUAL_COUNTER + 1
        else:
            learner, queried_val = apps.get_app_config("home").learn_ratings(CUR_QINSTANCE, client_list)
            print("cur_PREDS={}, queried_val={}, cur_qinstance={} @client_to_view".format(CUR_PREDS, queried_val, CUR_QINSTANCE))
            if apps.get_app_config("home").count == 0: # If no Blob rows exist, this will run the first time.
                f1, f2, bname = ORIGINALH5_ONE, ORIGINALH5_TWO, "originalModel"
            else:
                f1, f2, bname = MODIFIEDH5_ONE, MODIFIEDH5_TWO, "learnedModel"
            learner.save_model(f1, f2)
            blob = Blobby.objects.create(response=resp, name=bname)
            blob.set_data_one(f1)
            blob.set_data_two(f2)
            blob.save()

        AnswerVideo.objects.create(
            caption_title=CAPTION_TITLE, clip_title=VIDEO_TITLE,
            delay=queried_val[0][0], speed=queried_val[0][1], mw=queried_val[0][2], pv=queried_val[0][3],
            delay_pred=CUR_PREDS[0], speed_pred=CUR_PREDS[1], mw_pred=CUR_PREDS[2], pv_pred=CUR_PREDS[3],
            question=q, response=resp, category=category, body=client_list
        )
        WAIT_SWITCH = True
        print("Posted! WAIT_SWITCH IS NOW:", WAIT_SWITCH)
        return HttpResponse("success")
    else:
        print(request.method)
        return HttpResponse("success")
        

def noconsent(request):
    email = "somang@mie.utoronto.ca"
    return render(request, "noconsent.html", {"email": email})

def byebye(request):
    print("STUDY FINISHED WITH THIS PARTICIPANT.")
    email = "somang@mie.utoronto.ca"
    return render(request, "byebye.html", {"email": email})
