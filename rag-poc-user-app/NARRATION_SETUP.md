# Narration Feature Setup

## AWS Polly Configuration

To enable the narration feature, you need to set up Amazon Polly with your AWS credentials.

### Step 1: Create .env file

Create a `.env` file in the root directory of your project with the following content:

```env
REACT_APP_AWS_ACCESS_KEY_ID=your_access_key_here
REACT_APP_AWS_SECRET_ACCESS_KEY=your_secret_access_key_here
REACT_APP_AWS_REGION=us-east-1
```

### Step 2: AWS Credentials

1. Go to the [AWS Console](https://console.aws.amazon.com)
2. Navigate to IAM (Identity and Access Management)
3. Create a new user or use an existing user
4. Attach the `AmazonPollyReadOnlyAccess` policy to the user
5. Generate access keys for the user
6. Copy the Access Key ID and Secret Access Key to your `.env` file

### Step 3: Restart the Application

After creating the `.env` file, restart your React application:

```bash
npm start
```

## Features

The narration feature includes:

- **Play/Pause**: Click the play button to start narration, click pause to pause
- **Stop**: Stop the current narration and reset to beginning
- **Restart**: Restart narration from the beginning
- **Progress Indicator**: Shows current time and total duration
- **Error Handling**: Displays "Narration unavailable" if AWS is not configured properly

## Notes

- Narration only works for bot messages (not user messages)
- Audio is cached to avoid re-synthesizing the same text
- Uses Amazon Polly's neural voice engine with the "Joanna" voice
- Supports only one narration at a time - starting a new narration stops the current one