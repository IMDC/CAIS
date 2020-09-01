let triviaIndex = JSON.parse(document.getElementById('trivia-questions').textContent);

function handletrivia(){
    timeCounter();
    triviaFacts(triviaIndex[0]);
    var answer_text = $("#answerDiv").find("div").text().slice(-1);

    setTimeout(function(){        
        var option_text = $('input[name=options]:checked', '#trivia-question').val();
        if (answer_text === option_text){
            $("#trivia-answer").html() = "<b>Correct!</b>";
            $("#trivia-answer").css("color", "green");
            $("#answerDiv").toggleClass("visible");
        }else{
            $("#trivia-answer").css("color", "red");
            $("#answerDiv").toggleClass("visible");
        };
        setTimeout(function(){
            $("#answerDiv").toggleClass("visible");
            triviaFacts(triviaIndex[1]);
            var answer_text = $("#answerDiv").find("div").slice(-1);
            
            setTimeout(function(){
                var option_text = $('input[name=options]:checked', '#trivia-question').val();
                if (answer_text === option_text){
                    $("#trivia-answer").html() = "<b>Correct!</b>";
                    $("#trivia-answer").css("color", "green");
                    $("#answerDiv").toggleClass("visible");
                }else{
                    $("#trivia-answer").css("color", "red");
                    $("#answerDiv").toggleClass("visible");
                };
            }, 3000);

        }, 3000);

    }, 6000);
    
};

function modalHandler(element){
    element.dataset.target = "#ModalCenter";
    $('#ModalCenter').modal();
    $('#ModalCenter').show();
    $('#ModalCenter').on('shown.bs.modal', handletrivia());
};

function checkForm(){
    let formValidation = true;
    console.log("validateForm()->starsetBooleans: ");
    for (var i in starsetBooleans){
        console.log(starsetBooleans[i]);
        if(starsetBooleans[i] === false) {
            console.log(starsetBooleans);
            formValidation = false;
        }
    };
    if (formValidation){
        submitRatings();        
        let element = document.getElementById('submitRating');
        modalHandler(element);
        return true;
    }else{
        $('.alert').show();
        return false;
    };
}

$(document).ready(function(){
    //https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage
    // SessionStorage propery similar to localStorage, sessionStorage is cleared when the page session ends though.
    if (typeof window.sessionStorage != undefined){
        if(!sessionStorage.getItem('mySessionVal')){
            sessionStorage.setItem('mySessionVal', true);
            sessionStorage.setItem('storedWhen', Date.now());
        }
    }
});