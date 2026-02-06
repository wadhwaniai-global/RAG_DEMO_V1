from twilio.rest import Client

# Replace these with your actual API key SID and secret
api_key_sid = 'SK********************************'
api_key_secret = '********************************'

# Replace this with your Account SID (still required)
account_sid = 'AC********************************'

client = Client(api_key_sid, api_key_secret, account_sid)

# Send custom text message
message = client.messages.create(
    from_='whatsapp:+14155238886',  # Twilio Sandbox or approved number
    to='whatsapp:+918233869111',    # Recipient number
    body='Hi there! This is a custom WhatsApp message sent using Twilio API.'  # Custom message
)

print("Message SID:", message.sid)