// Collapsible
var coll = document.getElementsByClassName("collapsible");

for (let i = 0; i < coll.length; i++) {
    coll[i].addEventListener("click", function () {
        this.classList.toggle("active");

        var content = this.nextElementSibling;

        if (content.style.maxHeight) {
            content.style.maxHeight = null;
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
        }
    });
}

function getTime() {
    let today = new Date();
    hours = today.getHours();
    minutes = today.getMinutes();

    if (hours < 10) {
        hours = "0" + hours;
    }

    if (minutes < 10) {
        minutes = "0" + minutes;
    }

    let time = hours + ":" + minutes;
    return time;
}

function firstBotMessage() {
    let firstMessage = "Welcome to Imarisha Sacco,I'm here to help you with any query regarding our sacco?";
    document.getElementById("botStarterMessage").innerHTML = '<p class="botText"><span>' + firstMessage + '</span></p>';

    let time = getTime();
    $("#chat-timestamp").append(time);
    document.getElementById("userInput").scrollIntoView(false);
}

firstBotMessage();

function getResponse() {
    let userInput = $("#user-input").val();

    if (userInput == "") {
        userInput = "I love Code Palace!";
    }

    let userHtml = '<p class="userInput"><span>' + userInput + '</span></p>';
    $("#user-input").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("chat-bar-bottom").scrollIntoView(true);

    setTimeout(() => {
        getHardResponse(userInput);
    }, 1000);
}

function buttonSendText(sampleText) {
    let userHtml = '<p class="userInput"><span>' + sampleText + '</span></p>';
    console.log(sampleText)
    $("#user-input").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("chat-bar-bottom").scrollIntoView(true);
    getHardResponse(sampleText); // Directly call getHardResponse
}

function sendButton() {
    getResponse();
}

function heartButton() {
    buttonSendText("Heart clicked!");
}

// Press enter to send a message
$("#user-input").keypress(function (e) {
    if (e.which == 13) {
        getResponse();
    }
});

function getBotResponse(userInput) {
    let csrfToken = $('input[name=csrf_token]').val();

    $.ajax({
        url: '/',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ user_input: userInput, csrf_token: csrfToken }),
        success: function (response) {
            console.log(response);
//            handleResponse(response.user_input, response.response);
            // Process the response here
            let botHtml = '<p class="botText"><span>' + response.response + '</span></p>';
            $("#chatbox").append(botHtml);
            document.getElementById("chat-bar-bottom").scrollIntoView(true);
        },
        error: function (error) {
            console.error('Error in AJAX request:', error);
        }
    });
}

function getHardResponse(userInput) {
			    let response = getBotResponse(userInput);
			    console.log(response)
//			    let botHtml = '<p class="botText"><span>' + response +   '</span></p>';
//			    $("#chatbox").append(botHtml);
//			    document.getElementById("chat-bar-bottom").scrollIntoView(true);
			}
