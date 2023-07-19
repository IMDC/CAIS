function checkRatingForm(){
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
        let element = document.getElementById('submitButton');
        $('.loader-wrapper').addClass('is-active');
        return true;
    }else{
        $('.alert').show();
        return false;
    };
}

$(document).ready(function(){
    
    // $("#submitButton").click(function(){
    //     $('.loader-wrapper').addClass('is-active');
    // })

    //https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage
    // SessionStorage propery similar to localStorage, sessionStorage is cleared when the page session ends though.
    if (typeof window.sessionStorage != undefined){
        if(!sessionStorage.getItem('mySessionVal')){
            sessionStorage.setItem('mySessionVal', true);
            sessionStorage.setItem('storedWhen', Date.now());
        }
    }
});