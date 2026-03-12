# 🚀 Enable Online Mode (Real AI)

Your Voice AI system is working in **offline mode** (rule-based responses) because:
- OpenAI API key has exceeded quota (429 error)
- Ollama is not running

## ✅ Option 1: Ollama (FREE - Recommended)

Run powerful AI models locally on your computer - completely FREE!

### Step 1: Install Ollama
**Windows:**
```bash
# Download from: https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

**After installation, Ollama will start automatically**

### Step 2: Pull a model
```bash
# Open a new terminal and run:
ollama pull llama3.2:3b

# For better quality (requires more RAM):
ollama pull llama3
```

### Step 3: Verify Ollama is running
```bash
# Should show available models:
ollama list
```

### Step 4: Restart your Voice AI server
The system will automatically detect Ollama and use it!

**Check logs for:**
```
✅ [INFO] llm_engine: Ollama provider ready: model=llama3
✅ [INFO] llm_engine: LLM initialised: chain=['OpenAI', 'Ollama', 'RuleBased']
```

---

## 💳 Option 2: OpenAI API (Paid)

If you prefer cloud-based AI with the highest quality:

### Step 1: Add credits to your OpenAI account
1. Go to https://platform.openai.com/settings/organization/billing
2. Add payment method and credits ($5 minimum)

### Step 2: Check your current key
Your key in `.env`: `sk-proj-FsEm...YsAA`

### Step 3: Test the key
```bash
cd backend
python test_api_key.py
```

If it fails, get a new key from: https://platform.openai.com/api-keys

---

## 🔍 Check Current Mode

Your system always shows which AI provider is active:

**In browser console:**
- Look for provider badges: `OpenAI`, `Ollama`, or `RuleBased`

**In terminal logs:**
```bash
[INFO] llm_engine: LLM initialised: chain=['OpenAI', 'Ollama', 'RuleBased']
[INFO] audio_router: LLM ready. Active provider: OpenAI
```

**At runtime:**
- If OpenAI fails → tries Ollama
- If Ollama fails → uses RuleBased (offline mode)

---

## 🎯 Quick Start (Ollama)

**Complete installation in 2 minutes:**

```bash
# 1. Install Ollama
winget install Ollama.Ollama

# 2. Wait 30 seconds for service to start, then pull a model
ollama pull llama3.2:3b

# 3. Restart your Voice AI server
# Press Ctrl+C in the server terminal, then:
cd "e:\hackthons\a voice infraastructure\voice-ai-system\backend"
python main.py
```

**That's it!** Your system will now use Ollama for intelligent AI responses! 🎉

---

## 📊 Performance Comparison

| Provider | Cost | Speed | Quality | Offline | 
|----------|------|-------|---------|---------|
| **OpenAI** | $$ | Fast | ⭐⭐⭐⭐⭐ | ❌ |
| **Ollama** | FREE | Medium | ⭐⭐⭐⭐ | ✅ |
| **Rule-Based** | FREE | Instant | ⭐⭐ | ✅ |

---

## 🐛 Troubleshooting

**"Ollama: not reachable"**
```bash
# Check if Ollama service is running:
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

**"OpenAI: Error code 429"**
- Your API key has no credits
- Add credits at: https://platform.openai.com/settings/organization/billing

**"Rule-based fallback provider ready"**
- This is normal! It's your safety net
- Always works even when other providers fail
