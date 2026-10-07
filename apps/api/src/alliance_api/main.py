from fastapi import FastAPI

app = FastAPI(title="Alliance Briefing API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
