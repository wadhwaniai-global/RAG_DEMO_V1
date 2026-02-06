# Narration Feature Troubleshooting Guide

## Quick Start

The narration feature uses Amazon Polly to convert bot responses to speech. Follow these steps to enable it:

### 1. Configure AWS Credentials

Create or update the `.env` file in the `rag-poc-user-app` directory:

```env
REACT_APP_AWS_ACCESS_KEY_ID=your_actual_access_key_here
REACT_APP_AWS_SECRET_ACCESS_KEY=your_actual_secret_key_here
REACT_APP_AWS_REGION=us-east-1
```

### 2. Get AWS Credentials

1. Go to [AWS Console](https://console.aws.amazon.com)
2. Sign in or create an AWS account
3. Navigate to **IAM** (Identity and Access Management)
4. Click **Users** → **Create User**
5. Name the user (e.g., "polly-narration-user")
6. Attach the policy: **AmazonPollyReadOnlyAccess**
7. Click **Create access key** → **Application running outside AWS**
8. Copy the **Access Key ID** and **Secret Access Key**
9. Paste them into your `.env` file

### 3. Restart the Application

After updating the `.env` file:

```bash
# Stop the development server (Ctrl+C)
# Then restart it
npm start
```

**IMPORTANT**: React only reads environment variables at startup, so you must restart the server after changing `.env`.

## How to Use Narration

1. Send a message and wait for the bot response
2. Look for the audio icon (🔊) below each bot message
3. Click the audio icon to start narration
4. Use the controls:
   - **Play/Pause**: Start or pause narration
   - **Stop**: Stop and reset to beginning
   - **Restart**: Start from beginning

## Troubleshooting

### Issue: Clicking audio button does nothing

**Solution**: Check the browser console (F12) for errors:

1. Open Developer Tools (F12)
2. Go to the **Console** tab
3. Click the audio button
4. Look for error messages

**Common errors and fixes:**

- `AWS Polly not configured`: Your `.env` file is missing or has placeholder values
- `Missing environment variables`: Make sure you restarted the app after creating `.env`
- `Invalid security token`: Your AWS credentials are incorrect
- `Access denied`: Your IAM user doesn't have Polly permissions

### Issue: "Narration unavailable" message

This means AWS Polly could not be initialized. Check:

1. `.env` file exists in `rag-poc-user-app` directory
2. All three environment variables are set with real values (not placeholders)
3. You restarted the React app after creating/updating `.env`

### Issue: Narration starts but no sound

1. Check your device volume
2. Check browser permissions (allow audio)
3. Try a different browser
4. Check browser console for playback errors

### Issue: AWS credentials error

Error message: `InvalidAccessKeyId` or `SignatureDoesNotMatch`

**Solution**:
1. Verify you copied the credentials correctly (no extra spaces)
2. Make sure the IAM user has **AmazonPollyReadOnlyAccess** policy
3. Generate new access keys if needed
4. Update `.env` and restart the app

### Issue: Region not supported

Error message: `InvalidParameterException` or `EndpointConnectionError`

**Solution**:
- Use a supported region: `us-east-1`, `us-west-2`, `eu-west-1`, etc.
- Update `REACT_APP_AWS_REGION` in `.env`
- Restart the app

## Verifying Configuration

To check if AWS Polly is configured correctly:

1. Open browser console (F12)
2. Click the audio button on any bot message
3. Look for these console messages:
   - ✅ `Starting narration for message: ...`
   - ✅ `Synthesizing speech with AWS Polly...`
   - ✅ `Speech synthesis successful, audio cached`
   - ✅ `Starting audio playback...`
   - ✅ `Audio playback started successfully`

If you see error messages instead:
- `AWS Polly not configured`: Fix your `.env` file
- `AWS Polly error`: Check AWS credentials and permissions

## Testing Without AWS

If you don't want to use AWS Polly, you can disable the narration feature by removing the audio button from the UI, or implement an alternative text-to-speech solution using:

- Web Speech API (browser built-in, limited voices)
- Google Cloud Text-to-Speech
- Microsoft Azure Cognitive Services
- OpenAI TTS API

## Security Best Practices

1. **Never commit `.env` file to git** (it's in `.gitignore`)
2. **Use IAM user with minimal permissions** (Polly read-only)
3. **Rotate access keys regularly**
4. **Consider using temporary credentials** (STS) for production
5. **Use AWS Secrets Manager** for production deployments

## Cost Considerations

AWS Polly pricing (as of 2024):
- **Free tier**: 5 million characters per month for 12 months
- **After free tier**: $4.00 per 1 million characters (Standard voices)
- **Neural voices**: $16.00 per 1 million characters

For a typical chat application with 100-200 word responses:
- Average response: ~800 characters
- 5 million chars = ~6,250 responses
- Free tier covers most development/testing needs

## Additional Resources

- [AWS Polly Documentation](https://docs.aws.amazon.com/polly/)
- [IAM User Guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/)
- [AWS Free Tier](https://aws.amazon.com/free/)

## Need Help?

If you're still having issues:

1. Check all console logs (browser F12)
2. Verify AWS credentials in IAM console
3. Test credentials with AWS CLI: `aws polly describe-voices --region us-east-1`
4. Review the error message carefully
5. Ensure you restarted the React app after any `.env` changes
