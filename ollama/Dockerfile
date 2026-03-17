FROM ollama/ollama:latest

COPY init-ollama.sh /usr/local/bin/init-ollama.sh
RUN apt-get update && apt-get install -y curl && chmod +x /usr/local/bin/init-ollama.sh && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/init-ollama.sh"]