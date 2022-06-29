// Initialize butotn with users's prefered color
let changeColor = document.getElementById("changeColor");

// When the button is clicked, inject setPageBackgroundColor into current page
changeColor.addEventListener("click", async () => {
  //var action = document.getElementById("action");
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: runSpeechRecognition,
  });

});

function runSpeechRecognition() {
  //var textArea = document.getElementById('input-message');
  //textArea.value = "test test"; 
  var textarea = window.document.querySelector('textarea');
  //var action = document.getElementById("action");
  //action.innerHTML = "test test"; 
  // new speech recognition object
  var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
  var actionBox = document.createElement('div')
  actionBox.style["position"] = "fixed";
  actionBox.style["fontSize"] = "20px";
  actionBox.style["font-family"] = "georgia"
  actionBox.style["top"] = "95%";
  actionBox.style["left"] = "90.1%";
  actionBox.style["backgroundColor"] = "#FAFDF3";
  actionBox.style["transform"] = "translate(-50%,-50%)";
  actionBox.style["width"] = "15em";
  actionBox.style["height"] = "7em";
  actionBox.style["border"] = "1px solid #333";
  actionBox.style["padding"] = "8px 12px";
  document.body.append(actionBox)
  // This runs when the speech recognition service starts
  recognition.onstart = function() {
      //textarea.innerHTML = "<small>listening, please speak...</small>";
      actionBox.innerHTML = "<small>listening, please speak.</small>";
  };
  
  recognition.onspeechend = function() {
      //textarea.innerHTML = "<small>stopped listening, hope you are done...</small>";
      actionBox.innerHTML = "<small>stopped listening.</small>";
      recognition.stop();
  }

  // This runs when the speech recognition service returns result
  recognition.onresult = function(event) {
      var transcript = event.results[0][0].transcript;
      var confidence = event.results[0][0].confidence;
      textarea.innerHTML = transcript;
      actionBox.innerHTML = "<b>Text:</b> " + transcript + "<br/> <b>Confidence:</b> " + confidence*100+"%";
      //const getCaretPosition = e => e && e.selectionStart || -1;
      //var pos = getCaretPosition(document.getElementById('input-message')); 
      //actionBox.innerHTML = pos;
  };

  // start recognition
  recognition.start();
}