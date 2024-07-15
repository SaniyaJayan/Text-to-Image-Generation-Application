const inputTxt = document.getElementById("input")
const image = document.getElementById("image")
const button = document.getElementById("btn")

async function query() {
	image.src = "/loading.gif"
	const response = await fetch(
		"https://api-inference.huggingface.co/models/Melonie/text_to_image_finetuned",
		{
			headers: {
				Authorization: "Bearer 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",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify({"inputs": inputTxt.value}),
		}
	);
	const result = await response.blob();
	return result;
}

button.addEventListener('click', async function(){
    query().then((response) => {
	    const objectUrl = URL.createObjectURL(response)
        image.src = objectUrl
    });
})