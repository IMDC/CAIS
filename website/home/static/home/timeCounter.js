
const loading_time = 36;

function timeCounter() {
  let totalSeconds = 0;
  var secondsLabel = document.getElementById("secondsdp");

  setInterval(function() {
    ++totalSeconds;
    secondsLabel.innerText = Math.round(totalSeconds/loading_time*100) + "%";
    console.log(totalSeconds);
  }, 1000);
  
}


function countDowner(totalSeconds) {
  var secondsLabel = document.getElementById("secondsdp");
  secondsLabel.innerText = totalSeconds;
  var newCountdown = setInterval(function() {
    if (totalSeconds === 0) {
      clearInterval(newCountdown);
    }else{
      --totalSeconds;
    };
    secondsLabel.innerText = totalSeconds;
  }, 1000);
}