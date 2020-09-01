var starsetBooleans = {starset1: false, starset2: false, starset3: false, starset4: false};
let ratingsArray = new Array(4); let defaultRatingsArray = new Array(4);
let predictionDefault = true;
let jsonObject = JSON.parse(document.getElementById('prediction-data').textContent);
var csrftoken = getCookie('csrftoken');


machinePredictions = jsonObject.replace(/[\[\]]|"+/g,'').split(',');
console.log("NEW JSON OBJECT ", machinePredictions);
let starkeyiterator = 0; let starnameiterator = 0; let starvalue = 0; let starkey = 0;

var theRealValue = 0;
var theRealStar = '';
var modifiedStarset = {starset1: false, starset2: false, starset3: false, starset4: false};

let predictionVal; 

const predID = ["d_pred", "s_pred", "m_pred", "p_pred"]; 
const rateID = ["d_rate", "s_rate", "m_rate", "p_rate"];
const factors = ["", "delay", "speed", "missing words", "paraphrasing"];
const inputstar = ["inputstar1", "inputstar2", "inputstar3", "inputstar4"];
const starname = ["starset1", "starset2", "starset3", "starset4"];
const starsets = {starset1: "starset1", starset2: "starset2", starset3: "starset3", starset4: "starset4", starset5: "starset5"};
const pp = {starset1: "starrating1", starset2: "starrating2", starset3: "starrating3", starset4: "starrating4"};
const star_scale = {
    '0': "Click a circle to teach Cappy", 
    '1': "Strongly dissatisfied", 
    '2': "Dissatisfied", 
    '3': "Neither satisfied nor dissatisfied", 
    '4': "Satisfied", 
    '5': "Strongly satisfied"
};

$(document).ready(function () {
    // add video-end alert, to display the submit button ONLY when video was watched.
    var v = document.getElementsByTagName("video")[0];
    var b = document.getElementById("bottompane");
    v.addEventListener("ended", function() { 
        console.log('Video has been viewed!');
        b.style.visibility = "visible";
    }, true);

    // change the text of submit button when it reached to the end.
    let submitCount = document.getElementById('submit-count').textContent;
    let vidCount = document.getElementById('vid-count').textContent;
    document.getElementById("submitRating").innerHTML = (submitCount == vidCount)? "Submit": "Next";

    // ** Active button coloring script (I Agree / I Disagree) **
    $("div div div form div div button").click(function(e){
        $(this).css({"background-color": "#0275d8", "color":"white"})
        $(this).siblings("button").css({"background-color": "#e2e5de", "color":"black"})
    })
    // ** Active button coloring script END **
    
    // Creates each starList instance and places each in the appropriate div    
    for (var i = 0; i < 4; i++) {
        var createList = new createStarList(1);
        document.getElementById(inputstar[i]).append(createList);
    }
    for (let j = 0; j < machinePredictions.length; j++) {
        predictionVal = (machinePredictions[j].replace(/^\s+|\s+$/g, ''));
        defaultRatingsArray[j] = machinePredictions[j];
        document.getElementById(predID[j]).innerHTML = predictionVal;            
        document.getElementById(rateID[j]).innerHTML = star_scale[predictionVal];
    }

    let values = 0; let kstar = 0;
    values = machinePredictions;
    for (var rating in starsets) {
        $(`.${rating} .fa-star`).each(function (i, item) {
            if (i < values[kstar]) {
                $(item).addClass('checked')
            } else {
                $(item).removeClass('checked')
            }
        });
        kstar++;
    }
    $('.alert').hide();
});


function resetStars(starsets) {
    // When you click agree it finds the right starset and resets it.
    // Set the value of starset according to the starset selected
    let resetThisStar = starsets;    
    for (blacklist in starname) { // This should always be true for the current starset...
        if (starname[blacklist] == starsets) {
            var allElems = document.getElementsByName(starname[blacklist]);
            for (let i = 0; i < allElems.length; i++) {allElems[i].style.color = "black";}
        }
    }
    var k = 0;
    for (const property in pp) {
        if (`${property}` == resetThisStar){ratingsArray[k] = defaultRatingsArray[k];}
        k+=1;
    }
    console.log("ratingsArray is now: " + ratingsArray);
};

function explainingStarRating(starid, value){
    star_position = starid.slice(starid.length - 1);
    starlabel_dom = "starexplain" + star_position;
    if (value > 0){
        document.getElementById(starlabel_dom).innerHTML = "You are " + star_scale[value] + " with the " + factors[star_position];
    }else{
        document.getElementById(starlabel_dom).innerHTML = star_scale[value];
    }
        
}

function createStarList(value) {
    const starlist = document.createElement("div");
    starkey = [Object.keys(pp)[starkeyiterator]];
    
    // This is how to set new values for each key
    starvalue = (pp[starkey] = 0);
    starlist.customKey = starkey;
    starlist.customValue = starvalue;
    starlist.setAttribute('class', 'starset');
    starkeyiterator++;

    for (let i = 1; i < 6; i++) {
        const star = document.createElement("span");        
        star.className = "fa fa-circle-o";
        star.setAttribute("name", starname[starnameiterator]);        
        star.addEventListener("click", onStarClickListener);
        star.addEventListener("mouseover", onStarHoverListener);
        star.addEventListener("mouseout", onStarExitListener);        
        starlist.append(star);
    }
    starnameiterator++;
    return starlist;
}


function onStarExitListener(event) {
    const target = event.target;
    const sibilingStars = Array.from(target.parentNode.children);
    const targetIndex = sibilingStars.indexOf(target);

    let prev_clicked_idx = pp[target.parentNode.customKey];
    var new_idx = 0;

    if (prev_clicked_idx > 0){
        new_idx = prev_clicked_idx-1;
        sibilingStars.forEach((star, index) => {star.style.color = (index == targetIndex)? "rgb(31, 120, 255)": "black";});
        explainingStarRating(target.parentNode.customKey[0], new_idx+1);
    } else {
        sibilingStars.forEach((star, index) => {star.style.color = "black";});
        explainingStarRating(target.parentNode.customKey[0], new_idx);
    }
    if ($(event.target).hasClass("fa-circle")){
        $(event.target).toggleClass('fa-circle-o fa-circle');
    };
    
}   

function onStarHoverListener(event) {
    const target = event.target;
    const sibilingStars = Array.from(target.parentNode.children);
    const targetIndex = sibilingStars.indexOf(target);
    sibilingStars.forEach((star, index) => {star.style.color = (index == targetIndex)? "rgb(31, 120, 255)": "black";});    
    explainingStarRating(target.parentNode.customKey[0], targetIndex+1);
    
    $(event.target).toggleClass('fa-circle-o fa-circle');
    
}

function onStarClickListener(event) {
    let realValue = 0;
    const target = event.target;
    // find all sibiling stars
    const sibilingStars = Array.from(target.parentNode.children);
    // find the index of target sibiling
    const targetIndex = sibilingStars.indexOf(target);
    realValue = targetIndex + 1;
    // set starList value for caption rating
    pp[target.parentNode.customKey] = realValue;

    setRealValue(realValue, target.parentNode.customKey);
    // iterate sibilings, then color gold if index is <= clicked star, else black.
    sibilingStars.forEach((star, index) => {star.style.color = (index == targetIndex)? "rgb(31, 120, 255)": "black";});
    setCurrentStarRating(target.parentNode.customKey, realValue);
    explainingStarRating(target.parentNode.customKey[0], realValue);

}

function setCurrentStarRating(key, value) {
    predictionDefault = false;
    starsetBooleans[key[0]] = true;
    var k = 0;
    let currentStar = starsets;

    for (const property in pp) {
        if (parseInt(`${pp[property]}`) == 0) {
            ratingsArray[k] = machinePredictions[k];
        }
        if (key[0] == `${property}`){
            ratingsArray[k] = value;
        }
        k++;
    }
    console.log("ratingsarray is now: " + ratingsArray);
}

function updateUserRating(ratingsArr){    
    var ratingsArr = JSON.stringify(ratingsArr);   
    console.log("Rating submission:", ratingsArr);

    $.ajax({
        type: 'POST',
        url: "/client_to_view/",
        headers: { "X-CSRFToken": csrftoken },
        data: {client_id: ratingsArr}, // this passes the value to the views.py
        success: function( data ){
            console.log(data);
        }
    });    
}

function submitRatings(){
    var tmparray = (predictionDefault) ? defaultRatingsArray : ratingsArray;
    updateUserRating(tmparray);
}

function setRealValue(value, starset){ // Passing the value of index.js onStarClickListener's realvalue
    modifiedStarset[starset] = true;
    theRealValue = value;
    theRealStar = starset;
}


function showStarInput(value, starset){
    // Resets boolean value for starset key
    starsetBooleans[starset] = (modifiedStarset[starset] == true);
    var x = document.getElementById(value);
    void(x.style.visibility=="hidden" && (x.style.visibility = "visible"));
    explainingStarRating(starset, 0);
}

function hideStarInput(value, starset){
    starsetBooleans[starset] = true
    modifiedStarset[starset] = false
    let parentID = value.toString();
    $(`#${parentID} .starexplain`).empty();
    var x = document.getElementById(value);
    void(x.style.visibility=="visible" && (x.style.visibility = "hidden"));
}

