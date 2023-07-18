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

ORIGINALH5_ONE = str(settings.BASE_DIR) + "/originalfirst.h5"
ORIGINALH5_TWO = str(settings.BASE_DIR) + "/originalsecond.h5"
MODIFIEDH5_ONE = str(settings.BASE_DIR) + "/modifiedfirst.h5"
MODIFIEDH5_TWO = str(settings.BASE_DIR) + "/modifiedsecond.h5"

VIDCOUNT = 0
WAIT_SWITCH = False
CUR_PREDS = 0
CUR_QINSTANCE = 0
VIDEO_TITLE = ""
CAPTION_TITLE = ""
WATCHED_VIDEOS = []
VIDEO_POOL = []
VIDEO_CATEGORY = ""
UUID = ""
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
    max_vid_count = 5  # number of videos
    global MANUAL_IDX, VIDEO_POOL, LIST_NUMBERS, CUR_PREDS, CUR_QINSTANCE, VIDEO_TITLE, WATCHED_VIDEOS, CAPTION_TITLE

    if len(VIDEO_POOL) == 0:  # when video pool is empty
        return redirect(bye)
    else:
        if WAIT_SWITCH:
            print("WAIT SWITCH@index", WAIT_SWITCH)
            
            if apps.get_app_config("caisapp").count == 0:
                apps.get_app_config("caisapp").set_x_pool()
            
            ## 1. get prediction from cappy backend
            q_inst, preds, q_vals = apps.get_app_config(
                "caisapp"
            ).make_prediction()
            print(q_inst, preds, q_vals.astype(int))
            CUR_PREDS, CUR_QINSTANCE = preds, q_inst
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
            url = str(
                settings.STATICFILES_DIRS[0]
            ) + "/captions/base_captions/{}".format(
                VIDEO_TITLE.split("/")[1].split(".")[0] + "_0.vtt"
            )
            CaptionFile(url, q_vals)
            CAPTION_TITLE = (
                f"captions/{CaptionName.objects.last().caption_title}.vtt"
            )

            ## 5. pass the context values to initiate rendering
            context = {
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

        if apps.get_app_config("caisapp").count in MANUAL_IDX:
            print(
                "Learn Count={} MANUAL_IDX={}".format(
                    apps.get_app_config("caisapp").count, MANUAL_IDX
                )
            )
            queried_val = [[0, 0, 0, 0]]
            apps.get_app_config("caisapp").count = (
                apps.get_app_config("caisapp").count + 1
            )
            global MANUAL_COUNTER
            MANUAL_COUNTER = MANUAL_COUNTER + 1
        else:
            learner, queried_val = apps.get_app_config("caisapp").learn_ratings(
                CUR_QINSTANCE, client_list
            )
            print(
                "cur_PREDS={}, queried_val={}, cur_qinstance={} @client_to_view".format(
                    CUR_PREDS, queried_val, CUR_QINSTANCE
                )
            )
            if (
                apps.get_app_config("caisapp").count == 0
            ):  # If no Blob rows exist, this will run the first time.
                f1, f2, bname = ORIGINALH5_ONE, ORIGINALH5_TWO, "originalModel"
            else:
                f1, f2, bname = MODIFIEDH5_ONE, MODIFIEDH5_TWO, "learnedModel"
            learner.save_model(f1, f2)
            blob = Blobby.objects.create(response=resp, name=bname)
            blob.set_data_one(f1)
            blob.set_data_two(f2)
            blob.save()

        AnswerVideo.objects.create(
            caption_title=CAPTION_TITLE,
            clip_title=VIDEO_TITLE,
            delay=queried_val[0][0],
            speed=queried_val[0][1],
            mw=queried_val[0][2],
            pv=queried_val[0][3],
            delay_pred=CUR_PREDS[0],
            speed_pred=CUR_PREDS[1],
            mw_pred=CUR_PREDS[2],
            pv_pred=CUR_PREDS[3],
            question=q,
            response=resp,
            category=category,
            body=client_list,
        )
        WAIT_SWITCH = True
        print("Posted! WAIT_SWITCH IS NOW:", WAIT_SWITCH)
        return HttpResponse("success")
    else:
        print(request.method)
        return HttpResponse("success")


def bye(request):
    print("STUDY FINISHED WITH THIS PARTICIPANT.")
    email = "somang.nam[at]torontomu.ca"
    return render(request, "bye.html", {"email": email})
