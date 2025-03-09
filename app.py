from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from chatllm import chat_response


# تحميل المتغيرات من ملف .env
load_dotenv()

app = Flask(__name__)

# الحصول على API Key من المتغيرات البيئية
API_KEY = os.getenv('API_KEY')

@app.route('/chat', methods=['POST'])
def chat():
    # التحقق من وجود API Key في الطلب
    api_key = request.headers.get('API-Key')
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized! Invalid API Key."}), 403

    # الحصول على الرسالة من الطلب
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": 'Please provide a message!'}), 400
         
    response = chat_response(user_input)
    return jsonify({'response': response })    

if __name__ == '__main__':
    app.run(debug=True)
