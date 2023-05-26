 function triviaFacts(randomValue){
   
    $.getJSON('./../../static/trivia/wwtbam_qa.json', function(json){
        /**
         * json has {"question","A","B","C","D","answer"}
         */

        let question = json[randomValue].question;
        let A = json[randomValue].A;
        let B = json[randomValue].B;
        let C = json[randomValue].C;
        let D = json[randomValue].D;
        let answer = json[randomValue].answer;
        let options = {'A':A,'B':B,'C':C,'D':D};

        document.getElementById("trivia-category").innerHTML = "<b>Question:</b> " + question;

        var result = "";
        for (var i in options) {
            result += '<label class="triviaoptions">' + 
                        '<input type="radio" name="options" value=' + i + '></input>  ' + i + ': ' + options[i] + 
                    '</label>';
        }; 
        document.getElementById("trivia-question").innerHTML = result; // add buttons
        var nodes = document.getElementById("trivia-question").childNodes;
        
        
        document.getElementById("trivia-answer").innerHTML = "<b>Answer:</b> " + answer;
    });
};