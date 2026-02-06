from fastapi import FastAPI, Request
import uvicorn
from twilio.rest import Client
import httpx

app = FastAPI()
api_key_sid = 'SK********************************'
api_key_secret = '********************************'
account_sid = 'AC********************************'
client = Client(api_key_sid, api_key_secret, account_sid)

@app.post("/")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    from_number = form.get('From')
    to_number = form.get('To')
    message_body = form.get('Body')
    async with httpx.AsyncClient() as client:
        try:
            query_response = await client.post(
                "http://localhost:8000/query",
                json={
                    "query": message_body,
                    "use_query_expansion": True,
                    "use_reranking": True,
                    "document_filter": None,
                    "page_filter": None
                }
            )
            query_result = query_response.json()
            
            # Extract answer from response
            answer = query_result.get("answer", "Sorry, I couldn't process your query")


            # Send response back via WhatsApp
            message = client.messages.create(
                from_='whatsapp:+14155238886',
                to='whatsapp:+918233869111',  # Use the sender's number from the form
                body=answer
            )

            print(f"Response sent with Message SID: {message.sid}")

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            # Send error message back to user
            client = Client(api_key_sid, api_key_secret, account_sid)
            message = client.messages.create(
                from_='whatsapp:+14155238886',
                to='whatsapp:+918233869111',
                body="Sorry, I encountered an error processing your request."
            )
    return "OK"

@app.post("/status")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    from_number = form.get('From')
    message_body = form.get('Body')
    print(form)
    print(f"Message from {from_number}: {message_body}")
    return "OK"

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run("twilio-middleware:app", host="0.0.0.0", port=8888, reload=True)