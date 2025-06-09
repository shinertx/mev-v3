# ---
# role: [agent]
# purpose: Agent REST API
# dependencies: []
# mutation_ready: true
# test_status: [ci_passed]
# ---
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}
