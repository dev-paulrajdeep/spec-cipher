// async function send_to_flask() {
//     // const response = await fetch("http://127.0.0.1:5000/get_result");
//     const response = await fetch("/get_result");
//     const result = await response.text();

//     document.getElementById("btn-submit-analysis").textContent = result;
// }

async function send_to_flask() {

    const input =
        document.getElementById("spec-raw-text").value;

    const response = await fetch("/get_result", {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            user_input: input
        })
    });

    const result = await response.text();

    document.getElementById("btn-submit-analysis").textContent = result;
}