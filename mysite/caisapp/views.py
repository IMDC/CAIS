from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core import serializers
from django.apps import apps
from django.conf import settings
import time
import random
import json

from .caption_file import CaptionFile
from .models import (
    Question,
    AnswerVideo,
    Response,
    Category,
    VideoGenerator,
    Blobby,
    CaptionName,
)
from .forms import ResponseForm

# ORIGINALH5_ONE = settings.BASE_DIR + "/originalfirst.h5"
# ORIGINALH5_TWO = settings.BASE_DIR + "/originalsecond.h5"
# MODIFIEDH5_ONE = settings.BASE_DIR + "/modifiedfirst.h5"
# MODIFIEDH5_TWO = settings.BASE_DIR + "/modifiedsecond.h5"

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


def index(request):
    if request.method == "POST" or None:
        return redirect(survey)
    return render(request, "index.html")


def survey(request):
    form = ResponseForm(request.POST or None)
    context = {"form": form}
    global UUID, LIST_NUMBERS, MANUAL_IDX, VIDEO_POOL, VIDCOUNT, VIDEO_CATEGORY, WAIT_SWITCH
    UUID = form.uuid
    if form.is_valid():
        form.save()
        LIST_NUMBERS = list(range(0, 81))
        MANUAL_IDX = sorted(
            random.sample(range(2, 17), 2)
        )  # pick two numbers to insert the manual one in between..
        VIDEO_POOL = json.loads(
            serializers.serialize("json", VideoGenerator.objects.all())
        )
        VIDCOUNT = len(VIDEO_POOL)
        if Category.objects.filter(name="Video").count() < 1:
            Category.objects.create(name="Video")
        VIDEO_CATEGORY = Category.objects.filter(name="Video")[0].name
        WAIT_SWITCH = True
        return redirect(training)
    else:
        print(form.errors)
    return render(request, "survey.html", context)


def training(request):
    max_vid_count = 20  # 20 videos
    global MANUAL_IDX, VIDEO_POOL, LIST_NUMBERS, TRIVIA_QA, CUR_PREDS, CUR_QINSTANCE, VIDEO_TITLE, WATCHED_VIDEOS, CAPTION_TITLE

    if len(VIDEO_POOL) == 0:  # when video pool is empty
        return redirect(bye)
    else:
        if WAIT_SWITCH:
            print("WAIT SWITCH@index", WAIT_SWITCH)
            if apps.get_app_config("caisapp").count == 0:
                apps.get_app_config("caisapp").set_x_pool()  # initiate the dataset..?
            # 1. get prediction from cappy backend
            q_instance, preds, queried_vals = apps.get_app_config(
                "caisapp"
            ).make_prediction()

            print(q_instance, preds, queried_vals.astype(int))
            global CUR_PREDS
            CUR_PREDS = preds
            global CUR_QINSTANCE
            CUR_QINSTANCE = q_instance

            if (
                q_instance is None
            ):  # when there's no query_idx provided, we close the case
                return render(request, "byebye.html")
            # 2. parse the prediction passed from cappy.
            preds = list(map(lambda x: int(x), preds))
            preds = json.dumps(preds)
            # 3. load the video
            rand_video_sess = random.choice(VIDEO_POOL)
            global VIDEO_TITLE
            VIDEO_TITLE = rand_video_sess["fields"]["video_name"]
            VIDEO_POOL.remove(rand_video_sess)
            global WATCHED_VIDEOS
            WATCHED_VIDEOS.append(rand_video_sess)

            print(
                "VIDEO_POOL-length:{}, collection: {}".format(
                    len(VIDEO_POOL),
                    [vd["fields"]["video_name"] for vd in WATCHED_VIDEOS],
                )
            )
            #  pick the two numbers for the trivia questions..
            global LIST_NUMBERS
            rand_numbs = random.sample(LIST_NUMBERS, 2)
            global TRIVIA_QA
            TRIVIA_QA = rand_numbs
            LIST_NUMBERS.remove(rand_numbs[0])
            LIST_NUMBERS.remove(rand_numbs[1])
            if apps.get_app_config("caisapp").count == 0:
                # initiate the dataset
                apps.get_app_config("caisapp").set_x_pool()
            else:
                ## 1. get prediction from cappy backend
                q_instance, preds, queried_vals = apps.get_app_config(
                    "caisapp"
                ).make_prediction()
                print(q_instance, preds, queried_vals.astype(int))
                CUR_PREDS, CUR_QINSTANCE = preds, q_instance
                ## 2. parse the prediction passed from cappy.
                preds = list(map(lambda x: int(x), preds))
                preds = json.dumps(preds)
                ## 3. load the video
                rand_video_sess = random.choice(VIDEO_POOL)
                VIDEO_TITLE = rand_video_sess["fields"]["video_name"]
                VIDEO_POOL.remove(rand_video_sess)
                WATCHED_VIDEOS.append(rand_video_sess)
                print(
                    "VIDEO_POOL-length:{}, collection: {}".format(
                        len(VIDEO_POOL),
                        [vd["fields"]["video_name"] for vd in WATCHED_VIDEOS],
                    )
                )
                ## 4. get url to pass the CaptionFile obj
                url = settings.STATICFILES_DIRS[
                    0
                ] + "/captions/base_captions/{}".format(
                    VIDEO_TITLE.split("/")[1].split(".")[0] + "_0.vtt"
                )
                CaptionFile(url, queried_vals)
                CAPTION_TITLE = "captions/{}.vtt".format(
                    CaptionName.objects.last().caption_title
                )

            ## Finally pass the context values to initiate rendering
            context = {
                "triviaQuestions": TRIVIA_QA,
                "submitReady": apps.get_app_config("caisapp").count,
                "vid_count": 20,
                "preds": preds,
                "videourl": VIDEO_TITLE,
                "caption_title": CAPTION_TITLE,
            }
            return render(request, "training.html", context)
        else:
            time.sleep(5)
            return training(request)


def bye(request):
    print("STUDY FINISHED WITH THIS PARTICIPANT.")
    email = "somang.nam[at]torontomu.ca"
    return render(request, "bye.html", {"email": email})
