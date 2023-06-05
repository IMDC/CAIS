var submitBool = false;

$(document).ready(function () {
    $(".form-demog div div").click(function (e) {
        $("li").prev("li").removeClass("selected")
        $(this).addClass("selected").find("input[type='radio']").prop("checked", true);

        $(this).css("background-color", "#FFD76F");
        $(this).find("label").css("color", "black");
        $(this).siblings("div").css("background-color", "#e2e5de");
        $(this).siblings("div").find("label").css("color", "white");
    });
    //https://developer.mozilla.org/en-US/docs/Web/API/Window/sessionStorage
    // SessionStorage propery similar to localStorage, sessionStorage is cleared when the page session ends though.
    if (typeof window.sessionStorage != undefined) {
        if (!sessionStorage.getItem('mySessionVal')) {
            sessionStorage.setItem('mySessionVal', true);
            sessionStorage.setItem('storedWhen', Date.now());
        }
    }
});