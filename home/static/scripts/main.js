// main.js

$(document).ready(function () {
    // Handle form submission
    $('#chat-form').submit(function (e) {
        e.preventDefault(); // Prevent the default form submission

        // Get user input from the form
        var userInput = $('#user-input').val();
        let csrfToken = $('input[name=csrf_token]').val(); // csrf token extraction

        // Make an AJAX request to server
        $.ajax({
            url: '/bot',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ user_input: userInput, csrf_token: csrfToken }),
            success: function (response) {


                // Handle the response from the server
                handleResponse(response.user_input, response.response);
            },
            error: function (error) {
                console.error('Error in AJAX request:', error);
            }
        });

        // Clear the input field after submitting
        $('#user-input').val('');
    });

    // Function to handle the response and update the UI
    function handleResponse(userInput, modelResponse) {
        // Create a new chat bubble for the user input
        var newUserInputBubble = '<div class="chat-bubble me">' + userInput + '</div>';

        // Create a new chat bubble for the model response
        var newChatBubble = '<div class="chat-bubble you">' + modelResponse + '</div>';

        // Append the new chat bubbles to the chat body

        $('.chat-body').append(newUserInputBubble);
        $('.chat-body').append(newChatBubble);

        // Optionally, you can scroll the chat body to the bottom to show the latest message
        var chatBody = $('.chat-body');
        chatBody.scrollTop(chatBody[0].scrollHeight);

        // Call the updateChatUI function
        updateChatUI();
    }

    // Function to update the chat UI
    function updateChatUI() {
        // Optionally, you can scroll the chat body to the bottom to show the latest message
        var chatBody = $('.chat-body');
        chatBody.scrollTop(chatBody[0].scrollHeight);
    }
});
