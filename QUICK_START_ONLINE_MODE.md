# 🚀 Quick Start: Enable Online Mode (Real AI)

Your Voice AI system now has **4-tier fallback**:
1. **Groq API** (ultra-fast, free tier) ⚡
2. **OpenAI API** (highest quality, paid) 
3. **Ollama** (local AI, 100% free & offline)
4. **Rule-based** (always works, basic responses)

---

## ✅ **Option 1: Groq API (FASTEST - 2 minutes)**

**FREE** with generous limits! Ultra-fast inference.

### Step 1: Get your free Groq API key
1. Visit: https://console.groq.com/keys
2. Sign up (free)
3. Click "Create API Key"
4. Copy the key (starts with `gsk_...`)

### Step 2: Add to your `.env` file
```bash
# Open: e:\hackthons\a voice infraastructure\voice-ai-system\.env
# Add your key:
GROQ_API_KEY=gsk_your_key_here
```

### Step 3: Restart server
```bash
# Press Ctrl+C in server terminal, then:
cd "e:\hackthons\a voice infraastructure\voice-ai-system\backend"
python main.py
```

**Done!** You'll see:
```
✅ [INFO] llm_engine: Groq provider ready: model=llama-3.3-70b-versatile
✅ [INFO] llm_engine: LLM initialised: chain=['Groq', 'OpenAI', 'Ollama', 'RuleBased']
```

---

## ✅ **Option 2: Ollama (FREE - LOCAL AI)**

**100% free and offline** - no API keys needed!

### Current Status: 
Ollama is **downloading model now** (running in background terminal).

### When download completes:
```bash
# Verify model is ready:
ollama list

# Should show:
# NAME              SIZE     MODIFIED
# llama3.2:3b       2.0 GB   2 minutes ago
```

**Just restart your server** and Ollama will automatically activate!

---

## 📊 **Performance Comparison**

| Provider | Speed | Cost | Quality | Offline | API Key Required |
|----------|-------|------|---------|---------|------------------|
| **Groq** | ⚡⚡⚡⚡⚡ | FREE* | ⭐⭐⭐⭐ | ❌ | ✅ (free signup) |
| **OpenAI** | ⚡⚡⚡⚡ | $$ | ⭐⭐⭐⭐⭐ | ❌ | ✅ (paid) |
| **Ollama** | ⚡⚡⚡ | FREE | ⭐⭐⭐⭐ | ✅ | ❌ |
| **Rule-based** | ⚡⚡⚡⚡⚡ | FREE | ⭐⭐ | ✅ | ❌ |

*Groq free tier: 30 requests/min, 14,400 requests/day

---

## 🎯 **Recommended Setup**

**For demo/development:**
```bash
GROQ_API_KEY=your_groq_key     # Primary - fast & free
# Ollama as backup (local)
# Rule-based as final fallback
```

**For production:**
```bash
GROQ_API_KEY=your_groq_key     # High-volume fast inference
OPENAI_API_KEY=your_openai_key # Highest quality fallback
# Ollama as offline backup
# Rule-based as final fallback
```

---

## 🔍 **Verify Current Mode**

Check server logs:
```
[INFO] llm_engine: LLM initialised: chain=['Groq', 'OpenAI', 'Ollama', 'RuleBased']
[INFO] audio_router: LLM ready. Active provider: Groq
```

In browser:
- Watch for provider badges when AI responds
- `Groq`, `OpenAI`, `Ollama`, or `RuleBased`

---

## 🐛 **Troubleshooting**

**"Groq: no API key configured"**
- Add `GROQ_API_KEY` to `.env` file
- Restart server

**"Ollama: not reachable"**
```bash
# Start Ollama service:
ollama serve

# Or check if model is downloaded:
ollama list
```

**Current behavior: Fallback to Rule-based**
- This is normal when no AI provider is available
- System always works, just with basic responses
- Once you add Groq key or Ollama model, AI activates automatically!

---

## 📝 **Current Status**

✅ Groq support added  
✅ `groq` package installed  
⏳ Ollama model downloading (check background terminal)  
📝 Waiting for Groq API key in `.env`

**Next step:** Add Groq API key (2 minutes) → instant AI responses! 🎉
