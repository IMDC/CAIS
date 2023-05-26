var submitBool = false;

function submitMessage(){
    $('.alertSurvey').show();
}
function modalHandler(){
    $('.alertSurvey').hide();
    submitBool = true
    let element = document.getElementById('submitDemoq');
    element.dataset.target = "#ModalCenter";
    $('#ModalCenter').modal();
    $('#ModalCenter').show();
    element.disabled=true; 
    element.value='Sending…';
    $('#ModalCenter').on('shown.bs.modal', timeCounter());
}

$("form").on('submit', modalHandler);

$(document).ready(function(){
    $('.alertSurvey').hide();
    // on click to one of the answers, change the other answers to gray.
    $("div form div ul li").click(function(e){
        $("li").prev("li").removeClass("selected")
        $(this).addClass("selected").find("input[type='radio']").prop("checked", true);

        $(this).css("background-color", "#0275d8");
        $(this).find("label").css("color", "white");
        $(this).siblings("li").css("background-color", "#e2e5de");
        $(this).siblings("li").find("label").css("color", "black");
    });
        
    //https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage
    // SessionStorage propery similar to localStorage, sessionStorage is cleared when the page session ends though.
    if (typeof window.sessionStorage != undefined){
        if(!sessionStorage.getItem('mySessionVal')){
            sessionStorage.setItem('mySessionVal', true);
            sessionStorage.setItem('storedWhen', Date.now());
        }
    }
});