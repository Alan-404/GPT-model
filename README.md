# Generative Pre - Trained Transformer (GPT) Model
<b>Author:</b> Nguyen Duc Tri (Alan Nguyen) <br>
<b>Github:</b> https://github.com/Alan-404 <br>
<b>Linkedin: </b> https://www.linkedin.com/in/%C4%91%E1%BB%A9c-tr%C3%AD-nguy%E1%BB%85n-269845210/
<b>Reference: </b>Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (2018). <i>Improving Language Understanding by Generative Pre-Training.</i>

## <b>Architecture</b>
<img src="./assets/GPT.png"/>
<center>Credit:<i> Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (2018). Improving Language Understanding by Generative Pre-Training.</i></center>

## <b>Setup Environment</b>
1. Clone this repo: <code>git clone https://github.com/Alan-404/GPT-model.git</code>
2. CD into project: <code>cd GPT-model</code>
3. (Optional) Create Conda Environment: <code>conda create --name {YOUR_PROJECT_NAME}</code>
4. (Optional) Activation Conda Environment: <code>conda activate {YOUR_PROJECT_NAME}</code>
5. Install packages: <code>pip install requirements.txt</code>

## <b>Dataset Setup</b>
If you have a pair question and corresponding answer, the data sample is look like: <code>{question} <__sep__> {answer}</code>