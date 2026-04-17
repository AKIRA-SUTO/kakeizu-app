from flask import Flask, request, send_file
import pandas as pd
from io import BytesIO

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h2>家系図作成</h2>
    <form action="/generate" method="post">
        <textarea name="csv" rows="10" cols="50"></textarea><br>
        <button type="submit">作成</button>
    </form>
    """

@app.route("/generate", methods=["POST"])
def generate():
    csv_data = request.form["csv"]

    df = pd.read_csv(BytesIO(csv_data.encode()))

    # ここは仮：あなたの描画処理に置き換える
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, download_name="output.csv", as_attachment=True)

app.run(host="0.0.0.0", port=10000)