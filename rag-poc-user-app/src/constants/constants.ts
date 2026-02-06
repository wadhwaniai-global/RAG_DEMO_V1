import { DefaultChips } from '../types';

// Bot-specific question categories for auto-sequential flow
interface BotQuestionCategories {
  [category: string]: string[];
}

interface ChipMetadata {
  iconText: string;
  description: string;
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
}

interface BotConfiguration {
  intro: string;
  chips: string[];
  chipMetadata?: Record<string, ChipMetadata>;
  questionCategories?: BotQuestionCategories;
}

// Bot configurations mapped by bot ID
export const BOT_CONFIGURATIONS: Record<string, BotConfiguration> = {
  // Kenya Health Mission Bot
  '68bfe27544ecbdb0e3ff695b': {
    intro: "Welcome to the Kenya Health Mission. Here are some suggestions to get started with your questions:",
    chips: [
      "What is the strategic direction of kenya health mission?",
      "What are the key challenges in the health sector?",
      "What is the strategy to reduce maternal mortality?",
      "What is the strategy to reduce HIV transmission?"
    ],
    chipMetadata: {
      "What is the strategic direction of kenya health mission?": {
        iconText: "Strategy",
        description: "Kenya's health mission strategic direction",
        primaryColor: "#1e3a8a",
        secondaryColor: "#3b82f6",
        accentColor: "#60a5fa"
      },
      "What are the key challenges in the health sector?": {
        iconText: "Challenges",
        description: "Key health sector challenges",
        primaryColor: "#7c2d12",
        secondaryColor: "#ea580c",
        accentColor: "#fb923c"
      },
      "What is the strategy to reduce maternal mortality?": {
        iconText: "Maternal",
        description: "Maternal health strategies",
        primaryColor: "#831843",
        secondaryColor: "#db2777",
        accentColor: "#f472b6"
      },
      "What is the strategy to reduce HIV transmission?": {
        iconText: "HIV",
        description: "HIV prevention approaches",
        primaryColor: "#064e3b",
        secondaryColor: "#059669",
        accentColor: "#34d399"
      }
    },
    // No auto-sequential questions for Kenya bot
  },

  // ASHA Bot - TODO: Replace with actual ASHA bot ID
  '690395cbb3750101f0b11fdb': {
    intro: "Welcome to ASHA Saheli",
    chips: ["Maternal Health", "Newborn Health", "TB & Malaria"],
    chipMetadata: {
      "Maternal Health": {
        iconText: "MH",
        description: "Pregnancy care and maternal wellness",
        primaryColor: "#831843",
        secondaryColor: "#db2777",
        accentColor: "#f472b6"
      },
      "Newborn Health": {
        iconText: "NH",
        description: "Newborn care and infant health",
        primaryColor: "#1e3a8a",
        secondaryColor: "#3b82f6",
        accentColor: "#60a5fa"
      },
      "TB & Malaria": {
        iconText: "TM",
        description: "TB & Malaria prevention",
        primaryColor: "#064e3b",
        secondaryColor: "#059669",
        accentColor: "#34d399"
      }
    },
    questionCategories: {
      "Maternal Health": [
        "A pregnant woman asks what check-ups she needs and why. What is the recommended visit schedule?",
        "In the eighth month, a pregnant woman develops swelling of feet and face with severe headache and blurred vision. What should be done?"
      ],
      "Newborn Health": [
        "During a home visit you find a child with watery diarrhoea. What immediate advice will you give the caregiver?",
        "A 7-month-old is not gaining weight well. What exactly will you counsel the mother to change in complementary feeding?"
      ],
      "TB & Malaria": [
        "In the village, an elder has had a cough with sputum for over 3 weeks and evening fevers. What should be evaluated?",
        "In a malaria-prone village, parents ask how to protect their baby and what to do when fever starts. What prevention message will you emphasise for nighttime sleep, and what early actions will you advise when fever appears?"
      ]
    }
  }
};

// Default fallback configuration
const DEFAULT_BOT_CONFIG: BotConfiguration = {
  intro: "Welcome! How can I assist you today?",
  chips: [
    "Tell me about your services",
    "What can you help me with?",
    "Show me common questions"
  ]
};

// Helper function to get bot configuration
export const getBotConfiguration = (botId: string): BotConfiguration => {
  return BOT_CONFIGURATIONS[botId] || DEFAULT_BOT_CONFIG;
};

// Helper function to create DEFAULT_SELECTIONS for a specific bot
export const createDefaultSelections = (botId: string): DefaultChips => {
  const config = getBotConfiguration(botId);

  return {
    sender_id: botId,
    receiver_id: 'user',
    message: {
      text: '',
      confidence_score: null,
      sources: null,
      retrieval_metadata: null,
      processing_time: null,
      status: null
    },
    id: `default-chips-${botId}`,
    offset: 0,
    is_read: false,
    is_delivered: false,
    is_seen: false,
    is_deleted: false,
    is_archived: false,
    is_pinned: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    intro: config.intro,
    chips: config.chips,
    defaultChips: true
  };
};

// Legacy export for backward compatibility
export const DEFAULT_SELECTIONS: DefaultChips = createDefaultSelections('68bfe27544ecbdb0e3ff695b');
