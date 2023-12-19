1. Create venv for local setup
python -m venv venv

2. Activate the Virtual Environment
venv\Scripts\activate

3. Inside (venv) do
pip install -r ./requirements.txt 

4. Place you openAI key/Hugging face token in .env files
OPENAI_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
GITHUB_PERSONAL_ACCESS_TOKEN=

5. From the root directory run below command
streamlit run app.py