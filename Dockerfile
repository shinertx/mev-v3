# ---
# role: [infra]
# purpose: Container image
# dependencies: []
# mutation_ready: true
# test_status: [ci_passed]
# ---
FROM python:3.10-slim
WORKDIR /app
COPY . .
CMD ["bash", "ci_local.sh"]
