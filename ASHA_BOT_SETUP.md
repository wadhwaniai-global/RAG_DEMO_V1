# ASHA Bot Auto-Sequential Setup Guide

## Implementation Complete! ✅

The auto-sequential question flow has been implemented. Here's what's been done:

### Changes Made:

1. **constants.ts** - Bot-specific configurations with auto-sequential questions
2. **ChatBox.tsx** - Logic to detect category clicks and auto-send questions sequentially

## How to Complete Setup:

### Step 1: Get ASHA Bot ID

You need to find the actual user ID of your ASHA bot from MongoDB.

**Option A: Using MongoDB Shell**
```bash
# Connect to MongoDB
mongo mongodb://localhost:27017/fastapi_db

# Find ASHA bot
db.users.find({ user_type: "bot", name: /asha/i })
```

**Option B: Using API**
```bash
# Make sure middleware is running
cd rag-poc-middleware-app
python main.py

# Get all bots
curl http://localhost:8000/api/v1/users/bots
```

**Option C: Check middleware logs or database directly**

### Step 2: Update constants.ts

Open: `rag-poc-user-app/src/constants/constants.ts`

Replace `'ASHA_BOT_ID_PLACEHOLDER'` with the actual ASHA bot ID:

```typescript
export const BOT_CONFIGURATIONS: Record<string, BotConfiguration> = {
  // Kenya Health Mission Bot
  '68bfe27544ecbdb0e3ff695b': {
    // ... existing Kenya config
  },

  // ASHA Bot - Replace this line:
  'ACTUAL_ASHA_BOT_ID_HERE': {  // <-- PUT REAL ID HERE
    intro: "Welcome to ASHA Assistant. Select a topic:",
    chips: ["Maternal Health", "Newborn Health", "TB & Malaria"],
    // ... rest stays the same
  }
};
```

### Step 3: Test the Flow

1. **Start the backend:**
   ```bash
   cd waig-rag-poc
   python -m api.main
   ```

2. **Start the middleware:**
   ```bash
   cd rag-poc-middleware-app
   python main.py
   ```

3. **Start the frontend:**
   ```bash
   cd rag-poc-user-app
   npm start
   ```

4. **Test the flow:**
   - Login to the app
   - Select ASHA bot conversation
   - You should see welcome message with 3 category chips
   - Click "Maternal Health"
   - Q1 should be sent automatically → Bot answers
   - Q2 should be sent automatically after Q1 is answered → Bot answers
   - Done!

## How It Works:

### Flow Diagram:
```
User opens ASHA bot chat
  ↓
Welcome chips displayed automatically:
  [Maternal Health] [Newborn Health] [TB & Malaria]
  ↓
User clicks [Maternal Health]
  ↓
Q1 sent automatically: "A pregnant woman asks what check-ups..."
  ↓
Bot responds to Q1
  ↓
Q2 sent automatically (500ms delay): "In the eighth month, a pregnant woman..."
  ↓
Bot responds to Q2
  ↓
Sequence complete - user can continue normal chat
```

### Key Features:

✅ **Auto-show welcome chips** on chat start
✅ **Category detection** - clicks on category chips trigger sequential flow
✅ **Sequential auto-send** - wait for bot response before sending next question
✅ **Bot-specific configs** - Kenya bot keeps old behavior, ASHA bot has new flow
✅ **Clean state management** - resets when switching bots

## Troubleshooting:

### Issue: Welcome chips not showing
- Check that `agent_id` is valid
- Check browser console for errors
- Verify bot ID in `BOT_CONFIGURATIONS`

### Issue: Questions not auto-sending
- Check that category name exactly matches (case-sensitive)
- Check browser console for "Error starting sequential flow"
- Verify `questionCategories` is defined for bot

### Issue: Bot not answering
- Check that RAG backend is running
- Check that middleware worker is running
- Verify bot has access to relevant documents

## Adding More Bots:

To add another bot with auto-sequential flow:

```typescript
'NEW_BOT_ID_HERE': {
  intro: "Welcome to New Bot!",
  chips: ["Category A", "Category B"],
  questionCategories: {
    "Category A": [
      "Question 1 for Category A",
      "Question 2 for Category A"
    ],
    "Category B": [
      "Question 1 for Category B",
      "Question 2 for Category B"
    ]
  }
}
```

## Next Steps:

1. ✅ Find ASHA bot ID
2. ✅ Update `constants.ts` with real ID
3. ✅ Test the complete flow
4. ✅ Adjust question wording if needed
5. ✅ Add more categories/questions as required

---

**Questions?** Check the code comments in:
- `rag-poc-user-app/src/constants/constants.ts`
- `rag-poc-user-app/src/pages/Chatscreen/ChatBox.tsx`
