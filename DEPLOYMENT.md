# Production Deployment Guide

## Production-Ready Features

The Voice AI System includes production-grade features:

✅ **Security**
- CORS middleware (configurable origins)
- Trusted host protection
- Rate limiting (100 req/min per IP by default)
- Security headers
- API docs disabling option

✅ **Reliability**
- Graceful shutdown handling
- Health check endpoints (`/health`, `/readiness`)
- Global exception handler
- Automatic LLM fallback chain
- Connection pooling

✅ **Observability**
- Structured logging with timestamps
- Configurable log levels
- Access logs
- Request tracking
- Error reporting

✅ **Performance**
- Async/await throughout
- WebSocket streaming
- Model caching and sharing
- Lazy loading

---

## Environment Configuration

### Development
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_DOCS=true
DEBUG=true
CORS_ORIGINS=*
```

### Production
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_DOCS=false
DEBUG=false
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
TRUSTED_HOSTS=yourdomain.com,www.yourdomain.com
RATE_LIMIT_REQUESTS=50
ACCESS_LOG=true
WORKERS=4
```

---

## Deployment Options

### 1. Docker (Recommended)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python", "backend/main.py"]
```

Build and run:
```bash
docker build -t voice-ai:latest .
docker run -d \
  --name voice-ai \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  voice-ai:latest
```

### 2. Kubernetes

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-ai
  template:
    metadata:
      labels:
        app: voice-ai
    spec:
      containers:
      - name: voice-ai
        image: voice-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        envFrom:
        - secretRef:
            name: voice-ai-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: voice-ai-service
spec:
  selector:
    app: voice-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f deployment.yaml
```

### 3. Cloud Platforms

#### AWS (Elastic Beanstalk)
```bash
eb init voice-ai --platform python-3.11
eb create production --instance-type t3.medium
eb deploy
```

#### Google Cloud (Cloud Run)
```bash
gcloud run deploy voice-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

#### Azure (App Service)
```bash
az webapp up \
  --name voice-ai \
  --resource-group voice-ai-rg \
  --runtime "PYTHON:3.11" \
  --sku B2
```

### 4. Traditional Server (systemd)

Create `/etc/systemd/system/voice-ai.service`:
```ini
[Unit]
Description=Voice AI System
After=network.target

[Service]
Type=simple
User=voiceai
WorkingDirectory=/opt/voice-ai-system
Environment="PATH=/opt/voice-ai-system/.venv/bin"
EnvironmentFile=/opt/voice-ai-system/.env
ExecStart=/opt/voice-ai-system/.venv/bin/python backend/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-ai
sudo systemctl start voice-ai
```

---

## Reverse Proxy (Nginx)

Create `/etc/nginx/sites-available/voice-ai`:
```nginx
upstream voice-ai {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # WebSocket support
    location /ws/ {
        proxy_pass http://voice-ai;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # HTTP traffic
    location / {
        proxy_pass http://voice-ai;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

---

## Monitoring

### Health Checks

```bash
# Liveness (is service running?)
curl http://localhost:8000/health

# Readiness (can service handle requests?)
curl http://localhost:8000/readiness
```

### Logging

Production logs to stdout/stderr with structured format:
```
2026-03-11 22:47:01 [INFO] main: Voice AI System — starting up
2026-03-11 22:47:05 [INFO] audio_router: LLM ready. Active provider: OpenAI
```

Collect with:
- Docker: `docker logs -f voice-ai`
- Systemd: `journalctl -u voice-ai -f`
- Kubernetes: `kubectl logs -f deployment/voice-ai`

### Metrics (Optional)

Add Prometheus metrics:
```bash
pip install prometheus-fastapi-instrumentator
```

In `main.py`:
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## Security Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Set `ENABLE_DOCS=false` (hides /docs, /redoc)
- [ ] Configure `CORS_ORIGINS` with specific domains
- [ ] Configure `TRUSTED_HOSTS` with specific domains
- [ ] Use HTTPS in production (reverse proxy with Let's Encrypt)
- [ ] Store API keys in secrets manager (AWS Secrets, Azure Key Vault, etc.)
- [ ] Never commit `.env` file to git
- [ ] Enable firewall rules (allow only 80/443)
- [ ] Set up rate limiting (default: 100 req/min)
- [ ] Configure log rotation
- [ ] Set up monitoring and alerting
- [ ] Regular security updates (`pip install --upgrade`)

---

## Performance Tuning

### Resource Requirements

| Component | Min | Recommended |
|-----------|-----|-------------|
| CPU | 1 core | 2+ cores |
| RAM | 2 GB | 4+ GB |
| Storage | 5 GB | 10+ GB |

### Workers

For CPU-bound tasks (without GPU):
```bash
WORKERS=1  # Single process (recommended for models)
```

For I/O-bound tasks (with external APIs):
```bash
WORKERS=4  # Multiple workers
```

**Note:** Since models are loaded into memory, multiple workers will duplicate them. Use 1 worker for now unless you implement shared memory.

### Model Optimization

- Use `WHISPER_MODEL_SIZE=base` (faster) vs `large-v3` (more accurate)
- Use `WHISPER_DEVICE=cuda` if GPU available
- Enable Ollama for offline LLM (no API latency)
- Use HF Whisper mode (`ASR_MODE=hf`) to offload ASR

---

## Troubleshooting

### High Memory Usage
- Reduce `WHISPER_MODEL_SIZE` (try `tiny` or `base`)
- Use `ASR_MODE=hf` instead of local Whisper
- Limit `WORKERS=1`

### Slow Response
- Enable Ollama for local LLM (faster than OpenAI)
- Use `WHISPER_DEVICE=cuda` if GPU available
- Check network latency to OpenAI API

### WebSocket Disconnects
- Check Nginx proxy timeouts
- Verify firewall allows WebSocket upgrades
- Check client-side timeout settings

### Rate Limit Errors
- Increase `RATE_LIMIT_REQUESTS`
- Distribute load across multiple instances
- Use Redis for distributed rate limiting

---

## Backup and Recovery

### Database (None currently)
This system is stateless. Sessions are in-memory only.

### Models
Models are cached in:
- `~/.cache/torch/hub` (Silero VAD)
- `~/.cache/huggingface` (if using HF)
- Download once, then can work offline

### Configuration
Backup `.env` file securely (contains API keys)

---

## Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use sticky sessions for WebSocket connections
- Share models via NFS or S3 mounts

### Vertical Scaling
- Increase CPU/RAM for larger models
- Add GPU for faster inference
- Use faster storage (SSD/NVMe)

---

## Support

For production issues:
- Check logs: `/health` and `/readiness` endpoints
- Review error traces in application logs
- Monitor resource usage (CPU, RAM, network)
- Test fallback chains (OpenAI → Ollama → Rules)

GitHub: https://github.com/Sathvik2005/realtime-voice-ai-infrastructure-SniperThink
