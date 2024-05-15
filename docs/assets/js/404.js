document.addEventListener('DOMContentLoaded', function(){
    // typing effect
    var blank_msgs = document.querySelectorAll('p');
    var typing_speed = 50;

    blank_msgs.forEach((item, index) => {
        var raw_text = item.innerText;
        var i = 0;

        item.innerHTML = "";

        function typeWriter() {
            if (i < raw_text.length) {
                item.innerHTML += raw_text.charAt(i);
                i++;
                setTimeout(typeWriter, typing_speed);
            }
        }

        typeWriter();
    });
});